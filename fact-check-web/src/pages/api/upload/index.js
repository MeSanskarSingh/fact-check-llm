export const config = {
  api: {
    bodyParser: false, // important for file upload
  },
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ message: "Method not allowed" });
  }

  try {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    const response = await fetch(`${backendUrl}/process`, {
      method: "POST",
      body: req, // forward request directly
    });

    const data = await response.json();

    return res.status(200).json(data);
  } catch (error) {
    console.error("Upload error:", error);

    return res.status(500).json({
      message: "Failed to process file",
    });
  }
}