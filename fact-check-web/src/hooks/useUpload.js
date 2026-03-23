import { useState } from "react";
import { apiRequest } from "@/lib/apiClient";

export default function useUpload() {
  const [loading, setLoading] = useState(false);

  const upload = async (fileOrText) => {
    setLoading(true);

    try {
      const res = await apiRequest("/analyze", {
        method: "POST",
        body: JSON.stringify({ input: fileOrText }),
      });

      return res; // should include id
    } finally {
      setLoading(false);
    }
  };

  return { upload, loading };
}