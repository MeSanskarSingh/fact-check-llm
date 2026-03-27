"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/router";

import Navbar from "@/components/layout/Navbar";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";

export default function DashboardPage() {
  const router = useRouter();
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const data = localStorage.getItem("history");
    if (data) {
      setHistory(JSON.parse(data));
    }
  }, []);

  return (
    <main className="min-h-screen text-white pt-20 bg-linear-to-br from-[#050505] via-[#0a0a1a] to-[#050505]">
      
      <Navbar />

      <div className="max-w-5xl mx-auto px-6 py-10">
        
        {/* Title */}
        <h1 className="text-3xl font-bold mb-8">
          Your Dashboard
        </h1>

        {/* Empty State */}
        {history.length === 0 && (
          <div className="text-center text-gray-400">
            <p>No analysis history yet.</p>
            <Button
              className="mt-4"
              onClick={() => router.push("/upload")}
            >
              Start Analyzing
            </Button>
          </div>
        )}

        {/* History */}
        <div className="space-y-4">
          {history.map((item) => (
            <Card
              key={item.id}
              className="p-5 flex items-center justify-between hover:bg-gray-900 transition cursor-pointer"
              onClick={() => {
                localStorage.setItem("result", JSON.stringify(item));
                router.push(`/result/${item.id}`);
              }}
            >
              <div>
                <p className="text-gray-400 text-sm">
                  Analysis
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
                  {Math.round(item.confidence * 100)}%
                </p>
              </div>
            </Card>
          ))}
        </div>

        {/* CTA */}
        {history.length > 0 && (
          <div className="mt-10 text-center">
            <Button onClick={() => router.push("/upload")}>
              New Analysis
            </Button>
          </div>
        )}
      </div>
    </main>
  );
}