"use client";

import { FiImage, FiVideo, FiMusic, FiFileText } from "react-icons/fi";

const options = [
  { label: "Image", icon: FiImage },
  { label: "Video", icon: FiVideo },
  { label: "Audio", icon: FiMusic },
  { label: "Text File", icon: FiFileText },
];

export default function FileOptions({ onSelect }) {
  return (
    <div className="flex justify-center gap-6 mt-4">
      {options.map((item, index) => {
        const Icon = item.icon;

        return (
          <button
            key={index}
            onClick={() => onSelect(item.label.toLowerCase())}
            className="flex flex-col items-center gap-1 text-gray-400 hover:text-white transition cursor-pointer"
          >
            <div className="p-3 rounded-full bg-gray-800 hover:bg-gray-700">
              <Icon className="w-5 h-5" />
            </div>
            <span className="text-xs">{item.label}</span>
          </button>
        );
      })}
    </div>
  );
}