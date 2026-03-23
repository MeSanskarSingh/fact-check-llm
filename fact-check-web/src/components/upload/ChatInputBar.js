"use client";

import { useState } from "react";
import { FiSend } from "react-icons/fi";

export default function ChatInputBar({ onTextSubmit }) {
  const [text, setText] = useState("");

  const handleSubmit = () => {
    if (!text.trim()) return;
    onTextSubmit(text);
    setText("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault(); // prevent newline
      handleSubmit();
    }
  };

  return (
    <div className="w-full flex items-center gap-2 bg-gray-900 border border-gray-700 rounded-full px-4 py-2 mt-6">
      
      <input
        type="text"
        placeholder="Type text to fact-check..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        className="flex-1 bg-transparent outline-none text-white placeholder-gray-500"
      />

      <button
        onClick={handleSubmit}
        className="p-2 rounded-full bg-purple-600 hover:bg-purple-500 transition"
      >
        <FiSend className="w-4 h-4 text-white" />
      </button>
    </div>
  );
}