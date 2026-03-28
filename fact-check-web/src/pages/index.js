import Navbar from "@/components/layout/Navbar";
import HeroSection from "@/components/landing/HeroSection";
import { BubbleBackground } from "@/components/backgrounds/BubbleBackground";
import Head from "next/head";

export default function Home() {
  return (
    <>
      <Head>
        <title>FactCheckLLM</title>
      </Head>
      <main className="bg-black text-white min-h-screen">
      <BubbleBackground interactive>
        <Navbar />
        <HeroSection />
        
      </BubbleBackground>
    </main>
    </>
  );
}