"use client";

import { useRouter } from "next/router";
import {
    FiHome,
    FiUpload,
    FiFileText,
    FiLogOut,
} from "react-icons/fi";

export default function Sidebar() {
    const router = useRouter();

    const menu = [
        { label: "Dashboard", icon: FiHome, path: "/dashboard" },
        { label: "Upload", icon: FiUpload, path: "/upload" },
        { label: "Results", icon: FiFileText, path: "/dashboard" },
    ];

    return (
        <aside className="h-screen w-64 bg-black border-r border-gray-800 p-6 flex flex-col justify-between">

            {/* Top */}
            <div>
                <h2 className="text-xl font-bold mb-8 text-white">
                    FactCheckLLM
                </h2>

                <nav className="space-y-4">
                    {menu.map((item, index) => {
                        const Icon = item.icon;
                        return (
                            <button
                                key={index}
                                onClick={() => router.push(item.path)}
                                className="w-full flex items-center gap-3 px-4 py-2 rounded-lg text-gray-300 hover:bg-gray-800 hover:text-white transition"
                            >
                                <Icon className="w-5 h-5" />
                                {item.label}
                            </button>
                        );
                    })}
                </nav>
            </div>

            {/* Bottom */}
            <button
                onClick={() => console.log("Logout")}
                className="flex items-center gap-3 px-4 py-2 rounded-lg text-red-400 hover:bg-red-500/10 transition"
            >
                <FiLogOut className="w-5 h-5" />
                Logout
            </button>
        </aside>
    );
}