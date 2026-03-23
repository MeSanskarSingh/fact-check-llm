"use client";

import Navbar from "@/components/layout/Navbar";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import { useRouter } from "next/router";

export default function DashboardPage() {
  const router = useRouter();

  // Mock history data
  const history = [
    {
      id: "1",
      type: "Text",
      verdict: "Fake",
      confidence: 82,
    },
    {
      id: "2",
      type: "Image",
      verdict: "Real",
      confidence: 91,
    },
    {
      id: "3",
      type: "Video",
      verdict: "Fake",
      confidence: 76,
    },
  ];

  return (
    <main className="min-h-screen text-white pt-20 bg-gradient-to-br from-[#050505] via-[#0a0a1a] to-[#050505]">
      
      <Navbar />

      <div className="max-w-5xl mx-auto px-6 py-10">
        
        {/* Title */}
        <h1 className="text-3xl font-bold mb-8">
          Your Dashboard
        </h1>

        {/* History List */}
        <div className="space-y-4">
          {history.map((item) => (
            <Card
              key={item.id}
              className="p-5 flex items-center justify-between hover:bg-gray-900 transition cursor-pointer"
              onClick={() => router.push(`/result/${item.id}`)}
            >
              <div>
                <p className="text-gray-400 text-sm">
                  {item.type} Analysis
                </p>
                <h3
                  className={`font-semibold ${
                    item.verdict === "Fake"
                      ? "text-red-500"
                      : "text-green-500"
                  }`}
                >
                  {item.verdict}
                </h3>
              </div>

              <div className="text-right">
                <p className="text-gray-400 text-sm">
                  Confidence
                </p>
                <p className="font-semibold">
                  {item.confidence}%
                </p>
              </div>
            </Card>
          ))}
        </div>

        {/* CTA */}
        <div className="mt-10 text-center">
          <Button onClick={() => router.push("/upload")}>
            New Analysis
          </Button>
        </div>
      </div>
    </main>
  );
}