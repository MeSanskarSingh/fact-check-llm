import clsx from "clsx";

export default function Button({
  children,
  onClick,
  type = "button",
  variant = "primary",
  className = "",
  disabled = false,
}) {
  const baseStyles =
    "inline-flex items-center justify-center font-medium transition duration-200 focus:outline-none";

  const variants = {
    primary:
      "bg-linear-to-r from-purple-500 via-purple-600 to-purple-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-purple-300 dark:focus:ring-purple-800 shadow-lg shadow-purple-500/50 dark:shadow-lg dark:shadow-purple-800/80 hover:opacity-90 transition cursor-pointer",
    secondary:
      "bg-gray-800 text-white border border-gray-600 hover:bg-gray-700 cursor-pointer",
    outline:
      "border border-gray-500 text-white hover:bg-gray-800 focus:ring-gray-200 focus:ring-4",
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        baseStyles,
        variants[variant],
        "px-5 py-2.5 rounded-lg",
        disabled && "opacity-50 cursor-not-allowed",
        className
      )}
    >
      {children}
    </button>
  );
}