import { FiImage, FiVideo, FiMusic, FiFileText } from "react-icons/fi";
import Card from "../ui/Card";

const features = [
  { icon: FiImage, label: "Image Analysis" },
  { icon: FiVideo, label: "Video Verification" },
  { icon: FiMusic, label: "Audio Detection" },
  { icon: FiFileText, label: "Text Fact-Check" },
];

export default function FeatureHighlights() {
  return (
    <section className="py-16 px-6 max-w-5xl mx-auto">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {features.map((item, index) => {
          const Icon = item.icon;
          return (
            <Card
              key={index}
              className="flex flex-col items-center justify-center text-center p-6 hover:scale-105 transition"
            >
              <Icon className="w-8 h-8 mb-3 text-purple-400" />
              <p className="text-sm text-gray-300">{item.label}</p>
            </Card>
          );
        })}
      </div>
    </section>
  );
}