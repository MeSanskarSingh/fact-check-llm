import NextAuth from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import CredentialsProvider from "next-auth/providers/credentials";
// import { connectDB } from "@/lib/db";
// import User from "@/models/User";
// import bcrypt from "bcrypt";

export default NextAuth({
  providers: [
    // Google Login
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,

      profile(profile) {
        return {
          id: profile.sub,
          name: profile.name,
          email: profile.email,
          image: profile.picture,
        };
      },
    }),

    // Mock Credentials
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: {},
        password: {},
      },
      async authorize(credentials) {
        if (
          credentials.email === "test@example.com" &&
          credentials.password === "123456"
        ) {
          return {
            id: "1",
            name: "Test User",
            email: credentials.email,
          };
        }

        return null;
      },
    }),
  ],

  callbacks: {
  async jwt({ token, account, profile }) {
    // First time login (Google)
    if (account && profile) {
      token.picture = profile.picture;
      token.name = profile.name;
    }
    return token;
  },

  async session({ session, token }) {
    // Inject into session
    if (token) {
      session.user.image = token.picture;
      session.user.name = token.name;
    }
    return session;
  },
},

  session: {
    strategy: "jwt",
  },

  pages: {
    signIn: "/auth/login",
  },

  secret: process.env.NEXTAUTH_SECRET,
});