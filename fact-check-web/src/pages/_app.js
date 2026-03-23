import { useState, useEffect } from "react";
import { useRouter } from "next/router";
import { SessionProvider } from "next-auth/react";
import PageLoader from "@/components/ui/PageLoader";
import "@/styles/globals.css";

export default function App({ Component, pageProps: { session, ...pageProps } }) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const handleStart = () => setLoading(true);
    const handleStop = () => setLoading(false);

    router.events.on("routeChangeStart", handleStart);
    router.events.on("routeChangeComplete", handleStop);
    router.events.on("routeChangeError", handleStop);

    return () => {
      router.events.off("routeChangeStart", handleStart);
      router.events.off("routeChangeComplete", handleStop);
      router.events.off("routeChangeError", handleStop);
    };
  }, [router]);

  return (
    <SessionProvider session={session}>
      {loading && <PageLoader />}
      <Component {...pageProps} />
    </SessionProvider>
  );
}