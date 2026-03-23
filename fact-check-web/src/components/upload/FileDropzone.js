"use client";

import { useState, useRef } from "react";
import { FiUploadCloud } from "react-icons/fi";
import { getFileType } from "@/utils/fileHelpers";

export default function FileDropzone({ onFileSelect }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (!file) return;

    const type = getFileType(file);
    onFileSelect(file, type);
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const type = getFileType(file);
    onFileSelect(file, type);
  };

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDragEnter={() => setIsDragging(true)}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current.click()}
      className={`w-full p-10 border-2 border-dashed rounded-2xl text-center cursor-pointer transition
        ${
          isDragging
            ? "border-purple-500 bg-purple-500/10"
            : "border-gray-700 hover:border-purple-500"
        }`}
    >
      <FiUploadCloud className="mx-auto w-10 h-10 mb-4 text-purple-400" />

      <p className="text-gray-300">
        Drag & drop your file here, or{" "}
        <span className="text-purple-400">click to upload</span>
      </p>

      <p className="text-sm text-gray-500 mt-2">
        Supports image, video, audio, and text
      </p>

      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleChange}
      />
    </div>
  );
}