"use client";

import { FcGoogle } from "react-icons/fc";
import Button from "../ui/Button";

export default function GoogleButton({ onClick }) {
  return (
    <Button
      onClick={onClick}
      variant="secondary"
      className="w-full flex items-center justify-center gap-2"
    >
      <FcGoogle className="w-5 h-5" />
      Continue with Google
    </Button>
  );
}