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

  session: {
    strategy: "jwt",
  },

  pages: {
    signIn: "/auth/login",
  },

  secret: process.env.NEXTAUTH_SECRET,
});