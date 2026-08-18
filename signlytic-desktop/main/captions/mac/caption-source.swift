// macOS caption source.
//
// Prints one JSON line per change on stdout, exactly like the Windows
// PowerShell sidecar, so caption-stream.js consumes both the same way and
// caption-assembler.js needs no macOS special case.
//
// Why speech recognition rather than reading the system Live Captions window:
// that window can be read, through the Accessibility API, and tools/ax-probe.swift
// proves it. It is not used because doing so costs the user Accessibility
// permission, the most powerful on the machine, and requires them to switch
// Live Captions on themselves. Speech recognition is a public, permissioned
// API, runs on device, needs neither of those, and covers speech in the room
// directly. See the README for the full comparison.
//
//   --source mic       microphone, for conversation in the room
//   --source system    system audio, for calls and video, via ScreenCaptureKit
//
// Output records:
//   {"type":"status","state":"...","detail":"..."}
//   {"type":"caption","text":"<the whole rolling buffer>"}
//   {"type":"error","message":"..."}

import Foundation
import Speech
import AVFoundation
import ScreenCaptureKit
import CoreMedia

// Exit codes. caption-stream.js turns these into something a user can act on,
// so they must stay in step with _describeExit there.
let EXIT_ERROR = 1
let EXIT_NO_SPEECH_PERMISSION = 3
let EXIT_NO_MIC_PERMISSION = 4
let EXIT_NO_SCREEN_PERMISSION = 5
let EXIT_RECOGNISER_UNAVAILABLE = 6

// A recognition request is not allowed to run indefinitely, so it is recycled
// before it can be cut off mid-sentence. Finished text carries across, so the
// buffer stays continuous over a restart.
let RECOGNITION_RECYCLE_SECONDS = 50.0

// The Windows caption element holds the last few lines and lets older ones
// scroll away. This keeps the same shape rather than growing without bound,
// which also keeps the assembler's comparisons cheap.
let MAX_BUFFER_CHARS = 1200

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

// One queue for stdout so records can never interleave and produce a line that
// will not parse. The reader discards unparseable lines, so a torn write would
// silently drop a caption.
let outputQueue = DispatchQueue(label: "ai.signlytic.captions.output")

// Dev switch. When the app is started the way a user starts it, stdout goes
// nowhere, so there is no way to see what the helper actually reported. Set
// SIGNLYTIC_CAPTION_LOG to a file path to keep a copy.
let logPath = ProcessInfo.processInfo.environment["SIGNLYTIC_CAPTION_LOG"]

func emit(_ object: [String: Any]) {
    outputQueue.async {
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let line = String(data: data, encoding: .utf8) else { return }
        print(line)
        fflush(stdout)
        if let logPath = logPath, let bytes = (line + "\n").data(using: .utf8) {
            if let handle = FileHandle(forWritingAtPath: logPath) {
                handle.seekToEndOfFile()
                handle.write(bytes)
                try? handle.close()
            } else {
                try? bytes.write(to: URL(fileURLWithPath: logPath))
            }
        }
    }
}

func status(_ state: String, _ detail: String) {
    emit(["type": "status", "state": state, "detail": detail])
}

func errorLine(_ message: String) {
    emit(["type": "error", "message": message])
}

func fail(_ message: String, _ code: Int) -> Never {
    errorLine(message)
    // Give the output queue a moment to drain, or the reason for the exit is
    // lost and the user sees a bare code.
    outputQueue.sync { }
    exit(Int32(code))
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

var sourceKind = "mic"
var localeIdentifier = "en-GB"
var parentPid: pid_t = 0

var argIndex = 1
let args = CommandLine.arguments
while argIndex < args.count {
    let arg = args[argIndex]
    let next = argIndex + 1 < args.count ? args[argIndex + 1] : nil
    switch arg {
    case "--source":
        if let v = next { sourceKind = v; argIndex += 1 }
    case "--locale":
        if let v = next { localeIdentifier = v; argIndex += 1 }
    case "--parent-pid":
        if let v = next, let n = Int32(v) { parentPid = n; argIndex += 1 }
    default:
        break
    }
    argIndex += 1
}

guard sourceKind == "mic" || sourceKind == "system" else {
    fail("unknown source \(sourceKind), expected mic or system", EXIT_ERROR)
}

// ---------------------------------------------------------------------------
// Parent watch
// ---------------------------------------------------------------------------

// If the app is force-killed rather than shut down, this process would
// otherwise linger holding the microphone. The Windows sidecar takes the same
// precaution for the same reason.
func watchParent(_ pid: pid_t) {
    guard pid > 0 else { return }
    let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global())
    timer.schedule(deadline: .now() + 2, repeating: 2)
    timer.setEventHandler {
        if kill(pid, 0) != 0 {
            exit(0)
        }
    }
    timer.resume()
    // Held so the source is not released and cancelled immediately.
    parentTimer = timer
}
var parentTimer: DispatchSourceTimer?

// ---------------------------------------------------------------------------
// The rolling buffer
// ---------------------------------------------------------------------------

// Speech recognition reports the transcript of the current request, revised in
// place as it goes, which is the same behaviour the Windows caption element
// has. The one difference is that the transcript resets when a request is
// recycled, so finished text is kept here and prepended. The result is a
// single continuous buffer, which is what the assembler expects.
final class RollingBuffer {
    private let lock = NSLock()
    private var committed = ""
    private var live = ""
    private var lastChange = Date()

    func setLive(_ text: String) -> String {
        lock.lock(); defer { lock.unlock() }
        // Recognition does not always announce that an utterance is over. It
        // reports no final result and no error, and simply starts reporting a
        // new segment: "the weather is good today" is followed by "Thank",
        // which is not a revision of it. That silent reset is the end of an
        // utterance, and if it is not noticed the previous one is overwritten
        // and never signed. Observed, with both sentences recognised correctly
        // and neither ever reaching the avatar.
        //
        // Compared on words with punctuation and casing removed, because a
        // genuine revision does rewrite those after the fact: "hello the
        // weather" becomes "hello, the weather" and must not read as new speech.
        if !live.isEmpty && RollingBuffer.isNewSegment(after: live, text) {
            _ = commitLocked()
        }
        if text != live { lastChange = Date() }
        live = text
        return joinedLocked()
    }

    // Commits the current utterance once it has stopped changing. This is the
    // main way a sentence gets closed off, because recognition gives no final
    // result and no error when a speaker simply stops.
    //
    // Divergence alone is not enough to decide an utterance ended: recognition
    // revises heavily in place, and a revision that rewrites the opening words
    // looks exactly like a new segment. Committing on that produced fragments
    // like "Calligraphy hair." in testing, from speech that was really one
    // sentence still being corrected. Silence is the reliable signal.
    func commitIfSettled(_ now: Date, _ after: TimeInterval) -> String? {
        lock.lock(); defer { lock.unlock() }
        guard !live.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        guard now.timeIntervalSince(lastChange) >= after else { return nil }
        return commitLocked()
    }

    private static func words(_ text: String) -> [String] {
        let cleaned = text.lowercased().map { c -> Character in
            (c.isLetter || c.isNumber) ? c : " "
        }
        return String(cleaned).split(separator: " ").map(String.init)
    }

    // A new segment, rather than a revision of the current one. Requires both
    // that the text is not a continuation and that it got shorter: recognition
    // starts a fresh segment from its first word or two, whereas a revision
    // keeps roughly what it had or grows. Without the length test, rewriting
    // the opening words of a long sentence reads as a new one.
    static func isNewSegment(after previous: String, _ next: String) -> Bool {
        let a = words(previous)
        let b = words(next)
        if a.isEmpty || b.isEmpty { return false }
        let shared = min(a.count, b.count)
        let continues = Array(a.prefix(shared)) == Array(b.prefix(shared))
        return !continues && b.count < a.count
    }

    func commitLive(_ text: String) -> String {
        lock.lock(); defer { lock.unlock() }
        live = text
        return commitLocked()
    }

    // Called when the current recognition request ends for any reason. Speech
    // that was only ever a partial result would otherwise be thrown away: the
    // request is replaced, the new one starts from nothing, and a sentence the
    // user watched appear on screen simply vanishes. Observed exactly that,
    // with "The weather is good today" recognised in full and then dropped.
    func commitPending() -> String {
        lock.lock(); defer { lock.unlock() }
        return commitLocked()
    }

    // The end of a request means the speaker stopped, which is a sentence
    // boundary the Windows source never gets told about. It has to be turned
    // into punctuation, because the shared assembler splits on punctuation and
    // recognition does not reliably end a statement with a full stop: "the
    // weather is good today" comes back bare, and a bare tail reads as a
    // sentence still being spoken, so it would be held back for ever and never
    // signed. Questions came back with a question mark, which is what made this
    // look like it was working at first.
    private func commitLocked() -> String {
        var trimmed = live.trimmingCharacters(in: .whitespacesAndNewlines)
        live = ""
        if !trimmed.isEmpty {
            if let last = trimmed.last, !".!?".contains(last) {
                trimmed += "."
            }
            committed = committed.isEmpty ? trimmed : committed + " " + trimmed
            trimLocked()
        }
        return joinedLocked()
    }

    private func joinedLocked() -> String {
        let l = live.trimmingCharacters(in: .whitespacesAndNewlines)
        if committed.isEmpty { return l }
        if l.isEmpty { return committed }
        return committed + " " + l
    }

    // Drop the oldest text once the buffer is long enough, at a sentence
    // boundary where possible so a half sentence is never left at the front.
    private func trimLocked() {
        guard committed.count > MAX_BUFFER_CHARS else { return }
        let overflow = committed.count - MAX_BUFFER_CHARS
        let cutIndex = committed.index(committed.startIndex, offsetBy: overflow)
        var tail = String(committed[cutIndex...])
        if let boundary = tail.rangeOfCharacter(from: CharacterSet(charactersIn: ".!?")) {
            tail = String(tail[tail.index(after: boundary.lowerBound)...])
        }
        committed = tail.trimmingCharacters(in: .whitespaces)
    }
}

let buffer = RollingBuffer()
var lastEmitted = ""

// How long the transcript must stop changing before the utterance is treated as
// finished. Deliberately shorter than the assembler's own settle time, so the
// two do not stack into a noticeable delay before the avatar starts signing.
let COMMIT_AFTER_SILENCE = 1.0
var settleTimer: DispatchSourceTimer?

func publish(_ text: String) {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty, trimmed != lastEmitted else { return }
    lastEmitted = trimmed
    emit(["type": "caption", "text": trimmed])
}

// ---------------------------------------------------------------------------
// Recognition
// ---------------------------------------------------------------------------

final class Recogniser {
    private let recogniser: SFSpeechRecognizer
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var recycleTimer: DispatchSourceTimer?
    private let lock = NSLock()
    private var announcedListening = false
    private var restarting = false

    init(locale: String) {
        guard let r = SFSpeechRecognizer(locale: Locale(identifier: locale)) else {
            fail("speech recognition is not available for \(locale)", EXIT_RECOGNISER_UNAVAILABLE)
        }
        guard r.isAvailable else {
            fail("the speech recogniser for \(locale) is not available right now", EXIT_RECOGNISER_UNAVAILABLE)
        }
        recogniser = r
    }

    var usesOnDeviceRecognition: Bool { recogniser.supportsOnDeviceRecognition }

    func start() {
        lock.lock()
        let req = SFSpeechAudioBufferRecognitionRequest()
        // Without punctuation the assembler never sees a sentence end, so
        // nothing is ever released and the avatar stays still. This one line
        // is what makes the shared assembler work unchanged.
        req.addsPunctuation = true
        req.shouldReportPartialResults = true
        req.taskHint = .dictation
        // Captioning whatever the user can hear is not something to send to a
        // server when the machine can do it locally.
        if recogniser.supportsOnDeviceRecognition {
            req.requiresOnDeviceRecognition = true
        }
        request = req
        lock.unlock()

        task = recogniser.recognitionTask(with: req) { [weak self] result, error in
            guard let self = self else { return }
            if let result = result {
                let text = result.bestTranscription.formattedString
                if !self.announcedListening && !text.isEmpty {
                    self.announcedListening = true
                    status("attached", "listening")
                }
                if result.isFinal {
                    publish(buffer.commitLive(text))
                    // The task is finished now, not paused. Recognition stops
                    // dead here unless a new request is opened, and a final
                    // result arrives after every pause in speech, so this is
                    // the normal path rather than an edge case. Observed:
                    // without it only the first sentence spoken is ever signed.
                    self.beginRestart()
                } else {
                    publish(buffer.setLive(text))
                }
            }
            if let error = error {
                // Not shown to the user. Every one of these is transient and is
                // answered by opening a new request: "No speech detected" fires
                // whenever a room goes quiet, and a cancelled request is what a
                // deliberate restart looks like from the inside. Reporting them
                // put "recognition error: No speech detected" on screen under a
                // perfectly healthy avatar. Kept in the log for diagnosis.
                let ns = error as NSError
                emit([
                    "type": "debug",
                    "message": "recognition ended: \(error.localizedDescription)",
                    "domain": ns.domain,
                    "code": ns.code,
                ])
                self.beginRestart()
            }
        }

        scheduleRecycle()
    }

    // Recycled on a timer rather than waiting to be cut off, so the break
    // happens at a moment of our choosing and finished text is carried over.
    private func scheduleRecycle() {
        recycleTimer?.cancel()
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global())
        timer.schedule(deadline: .now() + RECOGNITION_RECYCLE_SECONDS)
        timer.setEventHandler { [weak self] in self?.beginRestart() }
        timer.resume()
        recycleTimer = timer
    }

    // One way in and out of a restart, whether it was asked for by the recycle
    // timer, by a finished result or by an error. Guarded, because a final
    // result and the timer can land at the same moment and starting two
    // requests would have both of them appending to the same buffer.
    private func beginRestart() {
        lock.lock()
        if restarting {
            lock.unlock()
            return
        }
        restarting = true
        let req = request
        let currentTask = task
        request = nil
        task = nil
        lock.unlock()

        recycleTimer?.cancel()
        // Ending the audio lets the recogniser finish what it has rather than
        // discarding a sentence still in flight.
        req?.endAudio()
        currentTask?.finish()

        // Keep whatever this request had recognised. Silence is what usually
        // ends a request, and silence after speech is exactly where a sentence
        // should be closed off, so this is the common path and not a salvage.
        publish(buffer.commitPending())

        lock.lock()
        restarting = false
        lock.unlock()
        // Started straight away rather than after a delay: every millisecond
        // between requests is speech that is not being listened to.
        start()
    }

    func append(_ pcm: AVAudioPCMBuffer) {
        lock.lock()
        let req = request
        lock.unlock()
        req?.append(pcm)
    }

    func stop() {
        recycleTimer?.cancel()
        lock.lock()
        let req = request
        let currentTask = task
        request = nil
        task = nil
        lock.unlock()
        req?.endAudio()
        currentTask?.cancel()
    }
}

// ---------------------------------------------------------------------------
// Audio sources
// ---------------------------------------------------------------------------

// Recognition wants a single channel. Both sources hand over whatever the
// hardware or the capture stream produces, so conversion happens in one place.
final class MonoConverter {
    private var converter: AVAudioConverter?
    private var inputFormat: AVAudioFormat?
    let outputFormat: AVAudioFormat

    init() {
        outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                     sampleRate: 16000,
                                     channels: 1,
                                     interleaved: false)!
    }

    func convert(_ input: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        if inputFormat != input.format || converter == nil {
            converter = AVAudioConverter(from: input.format, to: outputFormat)
            inputFormat = input.format
        }
        guard let converter = converter else { return nil }

        let ratio = outputFormat.sampleRate / input.format.sampleRate
        let capacity = AVAudioFrameCount(Double(input.frameLength) * ratio) + 1024
        guard let out = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return nil }

        var supplied = false
        var conversionError: NSError?
        let stat = converter.convert(to: out, error: &conversionError) { _, statusOut in
            if supplied {
                statusOut.pointee = .noDataNow
                return nil
            }
            supplied = true
            statusOut.pointee = .haveData
            return input
        }
        if stat == .error || out.frameLength == 0 { return nil }
        return out
    }
}

let converter = MonoConverter()

// Microphone. The straightforward case: the engine hands over buffers in the
// hardware's own format and they are converted like any other source.
final class MicrophoneSource {
    private let engine = AVAudioEngine()
    private let recogniser: Recogniser

    init(recogniser: Recogniser) {
        self.recogniser = recogniser
    }

    func start() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else {
            fail("no microphone input is available", EXIT_NO_MIC_PERMISSION)
        }
        input.installTap(onBus: 0, bufferSize: 2048, format: format) { [weak self] pcm, _ in
            guard let self = self, let mono = converter.convert(pcm) else { return }
            self.recogniser.append(mono)
        }
        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}

// System audio. This is what matches the Windows behaviour: it captures what
// the machine is playing, so a call or a video gets signed.
//
// ScreenCaptureKit is the only supported way to do this without asking the
// user to install a virtual audio device. It is a screen capture API, so it
// needs Screen Recording permission even though no picture is wanted, and the
// smallest legal video stream is configured alongside the audio.
@available(macOS 13.0, *)
final class SystemAudioSource: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let recogniser: Recogniser
    private let sampleQueue = DispatchQueue(label: "ai.signlytic.captions.audio")

    init(recogniser: Recogniser) {
        self.recogniser = recogniser
    }

    func start() async throws {
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        } catch {
            fail("macOS has not granted screen recording, which is how it allows system audio to be captured. Allow Signlytic AI under System Settings, Privacy and Security, Screen and System Audio Recording, then start captions again.", EXIT_NO_SCREEN_PERMISSION)
        }
        guard let display = content.displays.first else {
            fail("no display was found to capture system audio from", EXIT_ERROR)
        }

        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = 48000
        config.channelCount = 2
        // Otherwise the app would caption its own output if it ever made any.
        config.excludesCurrentProcessAudio = true
        // Audio is the point; the video stream is unavoidable, so make it as
        // small and as slow as the API will accept.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.queueDepth = 3

        let s = SCStream(filter: filter, configuration: config, delegate: self)
        try s.addStreamOutput(self, type: .audio, sampleHandlerQueue: sampleQueue)
        try await s.startCapture()
        stream = s
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        guard let pcm = Self.pcmBuffer(from: sampleBuffer) else { return }
        guard let mono = converter.convert(pcm) else { return }
        recogniser.append(mono)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        errorLine("system audio capture stopped: \(error.localizedDescription)")
        exit(Int32(EXIT_ERROR))
    }

    private static func pcmBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription),
              let format = AVAudioFormat(streamDescription: asbd) else { return nil }
        let frames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard frames > 0,
              let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return nil }
        pcm.frameLength = frames
        let copied = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer, at: 0, frameCount: Int32(frames), into: pcm.mutableAudioBufferList)
        guard copied == noErr else { return nil }
        return pcm
    }

    func stop() {
        stream?.stopCapture { _ in }
        stream = nil
    }
}

// ---------------------------------------------------------------------------
// Permissions
// ---------------------------------------------------------------------------

func requestSpeechPermission(_ done: @escaping () -> Void) {
    SFSpeechRecognizer.requestAuthorization { authStatus in
        switch authStatus {
        case .authorized:
            done()
        case .denied:
            fail("speech recognition permission was refused; allow it in System Settings, Privacy and Security, Speech Recognition", EXIT_NO_SPEECH_PERMISSION)
        case .restricted:
            fail("speech recognition is restricted on this Mac", EXIT_NO_SPEECH_PERMISSION)
        case .notDetermined:
            fail("speech recognition permission was not granted", EXIT_NO_SPEECH_PERMISSION)
        @unknown default:
            fail("speech recognition permission was not granted", EXIT_NO_SPEECH_PERMISSION)
        }
    }
}

func requestMicrophonePermission(_ done: @escaping () -> Void) {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        done()
    case .notDetermined:
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            if granted { done() }
            else { fail("microphone permission was refused; allow it in System Settings, Privacy and Security, Microphone", EXIT_NO_MIC_PERMISSION) }
        }
    default:
        fail("microphone permission was refused; allow it in System Settings, Privacy and Security, Microphone", EXIT_NO_MIC_PERMISSION)
    }
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

watchParent(parentPid)

// Nothing else closes off an utterance when a speaker just stops talking.
func startSettleTimer() {
    let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global())
    timer.schedule(deadline: .now() + 0.3, repeating: 0.3)
    timer.setEventHandler {
        if let text = buffer.commitIfSettled(Date(), COMMIT_AFTER_SILENCE) {
            publish(text)
        }
    }
    timer.resume()
    settleTimer = timer
}

status("starting", "preparing \(sourceKind == "mic" ? "microphone" : "system audio") captions")

let recogniser = Recogniser(locale: localeIdentifier)
var micSource: MicrophoneSource?
var systemSource: AnyObject?

// Stopped by caption-stream.js with a kill, so release the audio hardware
// rather than leaving the microphone indicator on until the process is reaped.
func shutDown() {
    micSource?.stop()
    if #available(macOS 13.0, *) { (systemSource as? SystemAudioSource)?.stop() }
    recogniser.stop()
    exit(0)
}

let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
termSource.setEventHandler { shutDown() }
termSource.resume()
signal(SIGTERM, SIG_IGN)

let intSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
intSource.setEventHandler { shutDown() }
intSource.resume()
signal(SIGINT, SIG_IGN)

startSettleTimer()

requestSpeechPermission {
    let onDevice = recogniser.usesOnDeviceRecognition
    status("starting", "speech recognition ready\(onDevice ? ", on device" : "")")

    if sourceKind == "mic" {
        requestMicrophonePermission {
            recogniser.start()
            let source = MicrophoneSource(recogniser: recogniser)
            do {
                try source.start()
            } catch {
                fail("could not start the microphone: \(error.localizedDescription)", EXIT_NO_MIC_PERMISSION)
            }
            micSource = source
            status("idle", "listening to the microphone")
        }
    } else {
        guard #available(macOS 13.0, *) else {
            fail("capturing system audio needs macOS 13 or later", EXIT_ERROR)
        }
        recogniser.start()
        let source = SystemAudioSource(recogniser: recogniser)
        systemSource = source
        Task {
            do {
                try await source.start()
                status("idle", "listening to system audio")
            } catch {
                fail("macOS has not granted screen recording, which is how it allows system audio to be captured. Allow Signlytic AI under System Settings, Privacy and Security, Screen and System Audio Recording, then start captions again.", EXIT_NO_SCREEN_PERMISSION)
            }
        }
    }
}

RunLoop.main.run()
