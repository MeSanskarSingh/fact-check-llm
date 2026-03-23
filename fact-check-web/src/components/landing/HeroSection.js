"use client";
import { useRouter } from "next/router";
import Button from "../ui/Button";
import FeatureHighlights from "./FeatureHighlights";

export default function HeroSection() {
  const router = useRouter();

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden text-white bg-gradient-to-br from-[#050505] via-[#0a0a1a] to-[#050505]">
      
      {/* Background Gradient Blobs */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-[-100px] left-[-100px] w-[400px] h-[400px] bg-purple-600 opacity-30 rounded-full blur-3xl"></div>
        <div className="absolute bottom-[-120px] right-[-100px] w-[400px] h-[400px] bg-pink-500 opacity-30 rounded-full blur-3xl"></div>
        <div className="absolute top-[40%] left-[60%] w-[300px] h-[300px] bg-blue-500 opacity-20 rounded-full blur-3xl"></div>
      </div>

      {/* Content */}
      <div className="max-w-4xl text-center px-6">
        
        {/* Heading */}
        <h1 className="text-4xl md:text-6xl font-bold leading-tight mt-20 mb-6">
          Upload Anything.
          <br />
          <span className="bg-gradient-to-r from-purple-400 via-pink-500 to-blue-500 text-transparent bg-clip-text">
            Verify Everything.
          </span>
        </h1>

        {/* Subtext */}
        <p className="text-gray-400 text-lg md:text-xl mb-8">
          Detect misinformation across images, videos, audio, and text — instantly.
        </p>

        {/* CTA */}
        <div className="flex justify-center">
          <Button
            onClick={() => router.push("/upload")}
            className="px-8 py-3 text-lg rounded-full bg-linear-to-r from-purple-500 via-purple-600 to-purple-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-purple-300 dark:focus:ring-purple-800 shadow-lg shadow-purple-500/50 dark:shadow-lg dark:shadow-purple-800/80 hover:opacity-90 transition"
          >
            Start Fact Checking
          </Button>
          
        </div>
        <FeatureHighlights />
      </div>
    </section>
    
  );
}