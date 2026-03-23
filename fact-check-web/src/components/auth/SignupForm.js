import { useState } from "react";
import Input from "../ui/Input";
import Button from "../ui/Button";
import GoogleButton from "./GoogleButton";

export default function SignupForm() {
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
  });

  const [errors, setErrors] = useState({});

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
  e.preventDefault();

  const res = await fetch("/api/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(form),
  });

  const data = await res.json();

  if (!res.ok) {
    alert(data.message);
  } else {
    router.push("/auth/login");
  }
};

  return (
    <form onSubmit={handleSubmit} className="space-y-5 w-full">
      
      <h2 className="text-2xl font-semibold text-center">Create Account</h2>

      <Input
        label="Full Name"
        type="text"
        name="name"
        value={form.name}
        onChange={handleChange}
        placeholder="Enter your name"
        error={errors.name}
      />

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
        placeholder="Create a password"
        error={errors.password}
      />

      <Button type="submit" className="w-full">
        Sign Up
      </Button>

      <div className="flex items-center gap-2">
        <div className="flex-1 h-px bg-gray-700" />
        <span className="text-sm text-gray-400">OR</span>
        <div className="flex-1 h-px bg-gray-700" />
      </div>

      <GoogleButton onClick={() => console.log("Google Signup")} />
    </form>
  );
}