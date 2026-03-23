"use client";

import { FiImage, FiVideo, FiMusic, FiFileText } from "react-icons/fi";
import Card from "../ui/Card";

export default function UploadPreview({ file, fileType }) {
  if (!file) return null;

  const renderPreview = () => {
    const url = URL.createObjectURL(file);

    switch (fileType) {
      case "image":
        return <img src={url} alt="preview" className="max-h-64 rounded-lg mx-auto" />;

      case "video":
        return (
          <video controls className="max-h-64 rounded-lg mx-auto">
            <source src={url} />
          </video>
        );

      case "audio":
        return (
          <audio controls className="w-full">
            <source src={url} />
          </audio>
        );

      case "text":
        return (
          <p className="text-gray-300 text-sm">
            {file.name}
          </p>
        );

      default:
        return (
          <p className="text-gray-400">
            Preview not available
          </p>
        );
    }
  };

  const getIcon = () => {
    switch (fileType) {
      case "image":
        return <FiImage className="w-5 h-5" />;
      case "video":
        return <FiVideo className="w-5 h-5" />;
      case "audio":
        return <FiMusic className="w-5 h-5" />;
      case "text":
        return <FiFileText className="w-5 h-5" />;
      default:
        return null;
    }
  };

  return (
    <Card className="mt-6 space-y-4">
      <div className="flex items-center gap-2 text-gray-300">
        {getIcon()}
        <span className="text-sm">{file.name}</span>
      </div>

      {renderPreview()}
    </Card>
  );
}