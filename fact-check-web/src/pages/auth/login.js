import LoginForm from "@/components/auth/LoginForm";
import Card from "@/components/ui/Card";
import Link from "next/link";
import { Particles } from "@/components/backgrounds/ParticlesBackground";
import { HiArrowLeft } from "react-icons/hi2";

export default function LoginPage() {
  return (
    <Particles className="bg-[#09000F]">
        <div className="absolute top-6 left-6 z-20">
            <Link 
            href="/" 
            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors duration-200 group"
            >
            <HiArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform duration-200" />
            <span className="text-sm font-medium">Back to Home</span>
            </Link>
        </div>
      {/* Foreground Content */}
      <div className="min-h-screen flex items-center justify-center text-white relative">
        
        <Card className="w-full max-w-md bg-black/50 backdrop-blur-md border border-gray-700">
          
          <LoginForm />

          <p className="text-sm text-gray-400 text-center mt-4">
            Don’t have an account?{" "}
            <Link href="/auth/signup" className="text-purple-400 hover:underline">
              Sign up
            </Link>
          </p>

        </Card>
      </div>

    </Particles>
  );
}