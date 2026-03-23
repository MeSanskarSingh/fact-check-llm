"use client";

import { useRouter } from "next/router";
import Navbar from "@/components/layout/Navbar";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";

export default function ResultPage() {
  const router = useRouter();
  const { id } = router.query;

  if (!id) {
  return (
    <main className="min-h-screen flex items-center justify-center text-white">
      Loading...
    </main>
  );
}

  // Mock Data (replace later with API)
  const result = {
    verdict: "Fake",
    confidence: 82,
    extractedText:
      "Breaking news: A massive event has shocked the world...",
    explanation:
      "The content contains multiple unverifiable claims and lacks credible sources. Similar patterns have been flagged in known misinformation datasets.",
  };

  return (
    <main className="min-h-screen text-white pt-20 bg-gradient-to-br from-[#050505] via-[#0a0a1a] to-[#050505]">
      
      <Navbar />

      <div className="max-w-4xl mx-auto px-6 py-10 space-y-6">
        
        {/* Title */}
        <h1 className="text-3xl font-bold">
          Analysis Result
        </h1>

        {/* Verdict Card */}
        <Card className="p-6 flex items-center justify-between">
          <div>
            <p className="text-gray-400">Verdict</p>
            <h2
              className={`text-2xl font-bold ${
                result.verdict === "Fake"
                  ? "text-red-500"
                  : "text-green-500"
              }`}
            >
              {result.verdict}
            </h2>
          </div>

          <div>
            <p className="text-gray-400">Confidence</p>
            <h2 className="text-xl font-semibold">
              {result.confidence}%
            </h2>
          </div>
        </Card>

        {/* Extracted Text */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-2">
            Extracted Content
          </h3>
          <p className="text-gray-300 leading-relaxed">
            {result.extractedText}
          </p>
        </Card>

        {/* AI Explanation */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-2">
            AI Explanation
          </h3>
          <p className="text-gray-300 leading-relaxed">
            {result.explanation}
          </p>
        </Card>

        {/* Actions */}
        <div className="flex gap-4">
          <Button onClick={() => router.push("/upload")}>
            Analyze Another
          </Button>

          <Button
            variant="secondary"
            onClick={() => router.push("/dashboard")}
          >
            Go to Dashboard
          </Button>
        </div>
      </div>
    </main>
  );
}