"use client"

import React, { useEffect, useRef } from "react"
import { cn } from "@/lib/utils"

export function UnderwaterBackground({
  className,
  children,
  intensity = 1,
  speed = 1,
}) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // We use window dimensions for the canvas to keep it fullscreen 
    // even as we scroll the container
    let width = window.innerWidth
    let height = window.innerHeight
    canvas.width = width
    canvas.height = height

    let animationId
    let tick = 0

    const particles = Array.from({ length: 40 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      size: 1 + Math.random() * 2,
      speed: 0.3 + Math.random() * 0.4,
      opacity: 0.4 + Math.random() * 0.4,
      wobbleOffset: Math.random() * Math.PI * 2,
    }))

    const handleResize = () => {
      width = window.innerWidth
      height = window.innerHeight
      canvas.width = width
      canvas.height = height
    }

    window.addEventListener('resize', handleResize)

    const animate = () => {
      tick += 0.02 * speed
      ctx.clearRect(0, 0, width, height)

      for (const p of particles) {
        p.y -= p.speed * speed
        p.x += Math.sin(tick * 1.5 + p.wobbleOffset) * 0.4

        if (p.y < -10) {
          p.y = height + 10
          p.x = Math.random() * width
        }

        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(180, 230, 255, ${p.opacity})`
        ctx.fill()
      }
      animationId = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      cancelAnimationFrame(animationId)
      window.removeEventListener('resize', handleResize)
    }
  }, [speed])

  const duration1 = 8 / speed
  const duration2 = 12 / speed
  const duration3 = 10 / speed

  return (
    // 1. Changed 'fixed' to 'relative' and added 'min-h-screen'
    <div
      ref={containerRef}
      className={cn("relative min-h-screen w-full", className)}
      style={{
        background: "linear-gradient(180deg, #010012 0%, #020026 40%, #000005 100%)",
      }}
    >
      {/* 2. Background Wrapper: Set to 'fixed' so visuals stay put while scrolling */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        {/* Caustic light layers */}
        <div className="absolute inset-0">
          <div
            className="absolute -inset-[50%] opacity-30"
            style={{
              background: `
                radial-gradient(ellipse 40% 30% at 30% 30%, rgba(100, 200, 255, ${0.4 * intensity}), transparent),
                radial-gradient(ellipse 35% 40% at 70% 40%, rgba(80, 180, 255, ${0.3 * intensity}), transparent),
                radial-gradient(ellipse 45% 35% at 50% 60%, rgba(120, 220, 255, ${0.35 * intensity}), transparent)
              `,
              animation: `caustic1 ${duration1}s ease-in-out infinite`,
              filter: "blur(40px)",
            }}
          />
          {/* ... other caustic divs stay the same ... */}
          <div
            className="absolute -inset-[50%] opacity-25"
            style={{
              background: `
                radial-gradient(ellipse 50% 40% at 60% 35%, rgba(150, 230, 255, ${0.35 * intensity}), transparent),
                radial-gradient(ellipse 40% 45% at 25% 55%, rgba(100, 200, 255, ${0.3 * intensity}), transparent)
              `,
              animation: `caustic2 ${duration2}s ease-in-out infinite`,
              filter: "blur(50px)",
            }}
          />
        </div>

        {/* Light rays */}
        <div className="absolute inset-0">
          {[0, 1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="absolute top-0"
              style={{
                left: `${15 + i * 18}%`,
                width: "8%",
                height: "100%",
                background: `linear-gradient(180deg, rgba(180, 230, 255, ${0.12 * intensity}) 0%, rgba(150, 210, 255, ${0.04 * intensity}) 50%, transparent 80%)`,
                transform: "skewX(-5deg)",
                animation: `ray ${6 + i * 2}s ease-in-out infinite`,
                animationDelay: `${i * -1.5}s`,
                filter: "blur(8px)",
              }}
            />
          ))}
        </div>

        {/* Particles canvas */}
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

        {/* Overlays (Vignette, Surface shimmer) */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_30%,transparent_0%,transparent_50%,rgba(0,20,40,0.6)_100%)]" />
      </div>

      {/* 3. Content layer: Standard flow, will dictate the page height */}
      {children && (
        <div className="relative z-10 w-full">
          {children}
        </div>
      )}

      <style jsx>{`
        @keyframes caustic1 {
          0%, 100% { transform: translate(0%, 0%) scale(1); }
          33% { transform: translate(5%, 3%) scale(1.05); }
          66% { transform: translate(-3%, -2%) scale(0.95); }
        }
        @keyframes caustic2 {
          0%, 100% { transform: translate(0%, 0%) scale(1); }
          50% { transform: translate(-6%, 4%) scale(1.08); }
        }
        @keyframes ray {
          0%, 100% { opacity: 0.6; transform: skewX(-5deg) translateX(0); }
          50% { opacity: 1; transform: skewX(-8deg) translateX(10px); }
        }
      `}</style>
    </div>
  )
}