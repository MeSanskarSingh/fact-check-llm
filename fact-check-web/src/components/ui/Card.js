import clsx from "clsx";

export default function Card({ children, className = "" }) {
  return (
    <div
      className={clsx(
        "bg-gray-900/70 backdrop-blur-md border border-gray-800",
        "rounded-2xl p-6 shadow-lg",
        className
      )}
    >
      {children}
    </div>
  );
}