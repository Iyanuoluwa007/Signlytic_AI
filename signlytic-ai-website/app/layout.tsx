import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";

export const metadata: Metadata = {
  title: "Signlytic AI — British Sign Language Translation",
  description:
    "Bidirectional BSL translation system. Upload BSL videos for English, convert English to BSL signing. 5,203 signs, 100% dictionary accuracy. Powered by Video-SWIN-T, Groq LLM, Coqui TTS.",
  keywords: [
    "BSL",
    "British Sign Language",
    "AI",
    "Translation",
    "Sign Language Recognition",
    "Accessibility",
    "Deep Learning",
  ],
  authors: [{ name: "Oke Iyanuoluwa Enoch" }],
  openGraph: {
    title: "Signlytic AI — BSL Translation",
    description:
      "Bidirectional BSL translation. 5,203 signs, 100% dictionary accuracy.",
    type: "website",
    locale: "en_GB",
    url: "https://signlytic-ai-website.vercel.app",
  },
  twitter: {
    card: "summary_large_image",
    title: "Signlytic AI — BSL Translation",
    description:
      "Translate between British Sign Language and English.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <Script
          src="https://www.googletagmanager.com/gtag/js?id=G-5V9VVTG2PE"
          strategy="afterInteractive"
        />
        <Script id="ga4-init" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'G-5V9VVTG2PE');
          `}
        </Script>
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}