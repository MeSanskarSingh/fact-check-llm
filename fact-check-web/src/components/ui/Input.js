"use client";

import { useState } from "react";
import clsx from "clsx";
import { FiEye, FiEyeOff } from "react-icons/fi";

export default function Input({
  label,
  type = "text",
  name,
  value,
  onChange,
  placeholder,
  error,
  className = "",
}) {
  const [showPassword, setShowPassword] = useState(false);

  const isPassword = type === "password";

  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm text-gray-300 mb-1">
          {label}
        </label>
      )}

      <div className="relative">
        <input
          type={isPassword && showPassword ? "text" : type}
          name={name}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          className={clsx(
            "w-full px-4 py-2 rounded-lg bg-gray-900 text-white border",
            "focus:outline-none focus:ring-2 focus:ring-purple-500",
            error ? "border-red-500" : "border-gray-700",
            isPassword && "pr-10", // space for icon
            className
          )}
        />

        {/* 👁 Toggle Button */}
        {isPassword && (
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
          >
            {showPassword ? (
              <FiEyeOff className="w-5 h-5" />
            ) : (
              <FiEye className="w-5 h-5" />
            )}
          </button>
        )}
      </div>

      {error && (
        <p className="text-sm text-red-500 mt-1">{error}</p>
      )}
    </div>
  );
}