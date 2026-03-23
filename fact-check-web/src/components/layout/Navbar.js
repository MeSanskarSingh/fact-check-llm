"use client";

import Link from "next/link";
import { useRouter } from "next/router";
import { useSession, signOut } from "next-auth/react";
import { FiUpload } from "react-icons/fi";
import Button from "../ui/Button";

export default function Navbar() {
  const router = useRouter();
  const { data: session } = useSession();

  return (
    <nav className="fixed top-0 left-0 w-full z-50 bg-black/40 backdrop-blur-md border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        
        {/* Logo */}
        <Link href="/" className="text-xl font-bold">
          <span className="bg-gradient-to-r from-purple-400 via-pink-500 to-blue-500 text-transparent bg-clip-text">
            FactCheckLLM
          </span>
        </Link>

        <div className="flex items-center gap-4">

          {/* If logged in */}
          {session ? (
            <div className="flex items-center gap-3">
              
              {/* Avatar */}
              <div className="flex items-center gap-2">
                {session.user?.image ? (
                  <img
                    src={session.user.image}
                    alt="avatar"
                    className="w-8 h-8 rounded-full"
                  />
                ) : (
                  <div className="w-8 h-8 rounded-full bg-purple-500 flex items-center justify-center text-sm">
                    {session.user?.name?.[0] || "U"}
                  </div>
                )}
              </div>

              {/* Logout */}
              <Button
                variant="secondary"
                onClick={() => signOut({ callbackUrl: "/" })}
              >
                Logout
              </Button>
            </div>
          ) : (
            <Button onClick={() => router.push("/auth/login")}>
              Login
            </Button>
          )}
        </div>
      </div>
    </nav>
  );
}