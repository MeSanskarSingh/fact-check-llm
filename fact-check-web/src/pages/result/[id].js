"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/router";

import Navbar from "@/components/layout/Navbar";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";

export default function ResultPage() {
  const router = useRouter();
  const { id } = router.query;

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load result from localStorage
  useEffect(() => {
    if (!router.isReady) return;

    const data = localStorage.getItem("result");

    if (data) {
      setResult(JSON.parse(data));
    }

    setLoading(false);
  }, [router.isReady]);

  // Loading state
  if (loading || !id) {
    return (
      <main className="min-h-screen flex items-center justify-center text-white">
        Loading result...
      </main>
    );
  }

  // No data fallback
  if (!result) {
    return (
      <main className="min-h-screen flex items-center justify-center text-white">
        <div className="text-center">
          <p className="mb-4 text-gray-400">No result found</p>
          <Button onClick={() => router.push("/upload")}>
            Go Back to Upload
          </Button>
        </div>
      </main>
    );
  }

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
              {Math.round(result.confidence * 100)}%
            </h2>
          </div>
        </Card>

        {/* Extracted Text */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-2">
            Extracted Content
          </h3>
          <p className="text-gray-300 leading-relaxed whitespace-pre-line">
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