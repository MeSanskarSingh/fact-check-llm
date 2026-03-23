import { useState } from "react";
import Input from "../ui/Input";
import Button from "../ui/Button";
import GoogleButton from "./GoogleButton";
import { signIn } from "next-auth/react";
import { useRouter } from "next/router";

export default function LoginForm() {
  const [form, setForm] = useState({
    email: "",
    password: "",
  });

  const [loading, setLoading] = useState(false);

  const router = useRouter();

  const [errors, setErrors] = useState({});

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    let newErrors = {};
    if (!form.email) newErrors.email = "Email is required";
    if (!form.password) newErrors.password = "Password is required";

    setErrors(newErrors);

    if (Object.keys(newErrors).length === 0) {
        setLoading(true);
        const res = await signIn("credentials", {
            email: form.email,
            password: form.password,
            redirect: false,
        });

        setLoading(false);

        if (res.error) {
        setErrors({ general: "Invalid credentials" });
        } else {
        router.push("/upload");
        }
    }
    };

  return (
    <form onSubmit={handleSubmit} className="space-y-5 w-full">
      
      <h2 className="text-2xl font-semibold text-center">Welcome Back</h2>

      <Input
        label="Email"
        type="email"
        name="email"
        value={form.email}
        onChange={handleChange}
        placeholder="Enter your email"
        error={errors.email}
      />

      <Input
        label="Password"
        type="password"
        name="password"
        value={form.password}
        onChange={handleChange}
        placeholder="Enter your password"
        error={errors.password}
      />

      {errors.general && (
        <p className="text-red-500 text-sm text-center">
            {errors.general}
        </p>
        )}

      <Button type="submit" className="w-full" disabled={loading}>
        {loading ? "Logging in..." : "Login"}
      </Button>

      <div className="flex items-center gap-2">
        <div className="flex-1 h-px bg-gray-700" />
        <span className="text-sm text-gray-400">OR</span>
        <div className="flex-1 h-px bg-gray-700" />
      </div>

      <GoogleButton onClick={() => signIn("google", { callbackUrl: "/upload" })} />
    </form>
  );
}