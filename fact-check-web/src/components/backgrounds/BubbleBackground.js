"use client";

import { motion, useMotionValue, useSpring } from "framer-motion";
import { useCallback, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

export function BubbleBackground({
  className,
  children,
  interactive = false,
  transition = { stiffness: 100, damping: 20 },
  colors = {
    first: "18,113,255",
    second: "221,74,255",
    third: "0,220,255",
    fourth: "200,50,50",
    fifth: "180,180,50",
    sixth: "140,100,255",
  },
}) {
  const containerRef = useRef(null);

  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const springX = useSpring(mouseX, transition);
  const springY = useSpring(mouseY, transition);

  const handleMouseMove = useCallback(
    (e) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      mouseX.set(e.clientX - centerX);
      mouseY.set(e.clientY - centerY);
    },
    [mouseX, mouseY]
  );

  useEffect(() => {
    if (!interactive) return;
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener("mousemove", handleMouseMove);
    return () => container.removeEventListener("mousemove", handleMouseMove);
  }, [interactive, handleMouseMove]);

  const makeGradient = (color) =>
    `radial-gradient(circle at center, rgba(${color}, 0.8) 0%, rgba(${color}, 0) 50%)`;

  return (
    <div
      ref={containerRef}
      className={cn(
        "fixed inset-0 overflow-hidden bg-gradient-to-br from-violet-950 to-blue-950",
        className
      )}
    >
      <div className="absolute inset-0" style={{ filter: "blur(40px)" }}>
        <motion.div
          className="absolute rounded-full mix-blend-hard-light"
          style={{
            width: "80%",
            height: "80%",
            top: "10%",
            left: "10%",
            background: makeGradient(colors.first),
          }}
          animate={{ y: [-50, 50, -50] }}
          transition={{ duration: 30, repeat: Infinity }}
        />

        {interactive && (
          <motion.div
            className="absolute rounded-full mix-blend-hard-light opacity-70"
            style={{
              width: "100%",
              height: "100%",
              background: makeGradient(colors.sixth),
              x: springX,
              y: springY,
            }}
          />
        )}
      </div>

      {children && <div className="relative z-10">{children}</div>}
    </div>
  );
}