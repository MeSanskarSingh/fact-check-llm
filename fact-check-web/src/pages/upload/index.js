"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/router";
import { useSession } from "next-auth/react";

import Head from "next/head";

import Navbar from "@/components/layout/Navbar";
import Button from "@/components/ui/Button";
import { UnderwaterBackground } from "@/components/backgrounds/UnderwaterBackground";

import UploadPreview from "@/components/upload/UploadPreview";
import ChatInputBar from "@/components/upload/ChatInputBar";
import FileOptions from "@/components/upload/FileOptions";
import { uploadFile } from "@/services/uploadService";

import { getFileType } from "@/utils/fileHelpers";

export default function UploadPage() {
  const router = useRouter();
  const fileInputRef = useRef(null);

  const { data: session, status } = useSession();

  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null);

  // Auth Guard
  useEffect(() => {
    if (status === "loading") return;

    if (!session) {
      router.replace("/auth/login");
    }
  }, [session, status, router]);

  // Loading State
  if (status === "loading") {
    return (
      <div className="min-h-screen flex items-center justify-center bg-black text-white">
        Loading...
      </div>
    );
  }

  // Prevent rendering if not authenticated
  if (!session) return null;

  // File selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    const type = getFileType(selectedFile);
    setFile(selectedFile);
    setFileType(type);
  };

  // Text input
  const handleTextSubmit = (text) => {
    const textFile = new File([text], "input.txt", { type: "text/plain" });

    setFile(textFile);
    setFileType("text");
  };

  // File option click
  const handleFileOption = () => {
    fileInputRef.current.click();
  };

  // Analyze
  const handleAnalyze = async () => {
    if (!file) return;

    try {
      const result = await uploadFile(file);

      // store current result
      localStorage.setItem("result", JSON.stringify(result));

      // store history
      const history = JSON.parse(localStorage.getItem("history")) || [];

      history.unshift({
        id: Date.now(),
        verdict: result.verdict,
        confidence: result.confidence,
        extractedText: result.extractedText,
        createdAt: new Date().toISOString(),
      });

      localStorage.setItem("history", JSON.stringify(history));

      router.push(`/result/${Date.now()}`);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <>
      <Head>
        <title>Upload and Analyze</title>
      </Head>

      <UnderwaterBackground>
        <main className="min-h-screen text-white pt-20 relative overflow-hidden">
          
          <Navbar />

          <div className="max-w-3xl mx-auto px-6 py-24 text-center">
            
            {/* Title */}
            <h1 className="text-3xl md:text-4xl font-bold mb-8">
              Upload Anything
            </h1>

            {/* Chat Input */}
            <ChatInputBar onTextSubmit={handleTextSubmit} />

            {/* File Options */}
            <FileOptions onSelect={handleFileOption} />

            {/* Hidden File Input */}
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              onChange={handleFileChange}
            />

            {/* Preview */}
            <UploadPreview file={file} fileType={fileType} />

            {/* Analyze Button */}
            {file && (
              <div className="flex justify-center mt-6">
                <Button
                  onClick={handleAnalyze}
                  className="px-8 py-3 text-lg rounded-full"
                >
                  Analyze
                </Button>
              </div>
            )}
          </div>
        </main>
        </UnderwaterBackground>
    </>
  );
}