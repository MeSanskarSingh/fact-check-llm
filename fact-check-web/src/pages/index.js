import Navbar from "@/components/layout/Navbar";
import HeroSection from "@/components/landing/HeroSection";
import { BubbleBackground } from "@/components/backgrounds/BubbleBackground";

export default function Home() {
  return (
    <main className="bg-black text-white min-h-screen">
      <BubbleBackground interactive>
        <Navbar />
        <HeroSection />
        
      </BubbleBackground>
    </main>
  );
}