"use client";

export default function PageLoader() {
  return (
    <div className="fixed inset-0 z-[999] flex items-center justify-center bg-black/80 backdrop-blur-md">
      <div className="w-10 h-10 border-4 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
    </div>
  );
}