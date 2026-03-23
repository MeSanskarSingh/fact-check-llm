import { useState, useEffect } from "react";
import { getToken, setToken, removeToken } from "@/lib/auth";
import { apiRequest } from "@/lib/apiClient";

export default function useAuth() {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const token = getToken();
    if (token) {
      setUser({ token }); // later fetch real user
    }
  }, []);

  const login = async (email, password) => {
    const data = await apiRequest("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });

    setToken(data.token);
    setUser(data.user);
  };

  const logout = () => {
    removeToken();
    setUser(null);
  };

  return { user, login, logout };
}