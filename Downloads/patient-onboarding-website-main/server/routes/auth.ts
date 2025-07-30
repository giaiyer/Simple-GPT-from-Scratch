import { RequestHandler } from "express";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { getDatabase } from "../database";

// JWT secret (in production, this should be an environment variable)
const JWT_SECRET = process.env.JWT_SECRET || "healthcare-secret-key";

interface RegisterRequest {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  dateOfBirth: string;
  gender: string;
  password: string;
  bloodType?: string;
  allergies?: string;
  currentMedications?: string[];
  insuranceProvider?: string;
  policyNumber?: string;
  groupNumber?: string;
}

interface LoginRequest {
  email: string;
  password: string;
}

export const registerUser: RequestHandler = async (req, res) => {
  try {
    const {
      firstName,
      lastName,
      email,
      phone,
      dateOfBirth,
      gender,
      password,
      bloodType,
      allergies,
      currentMedications = [],
      insuranceProvider,
      policyNumber,
      groupNumber
    }: RegisterRequest = req.body;

    // Validate required fields
    if (!firstName || !lastName || !email || !phone || !dateOfBirth || !gender || !password) {
      return res.status(400).json({
        error: "Missing required fields",
        details: "firstName, lastName, email, phone, dateOfBirth, gender, and password are required"
      });
    }

    // Validate email format
    const emailRegex = /\S+@\S+\.\S+/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({
        error: "Invalid email format"
      });
    }

    // Validate password strength
    if (password.length < 8) {
      return res.status(400).json({
        error: "Password must be at least 8 characters long"
      });
    }

    const db = await getDatabase();

    // Check if user already exists
    const existingUser = await db.get(
      "SELECT id FROM users WHERE email = ?",
      email.toLowerCase()
    );

    if (existingUser) {
      return res.status(409).json({
        error: "User already exists",
        message: "A user with this email address already exists. Please try logging in instead."
      });
    }

    // Hash password
    const saltRounds = 12;
    const passwordHash = await bcrypt.hash(password, saltRounds);

    // Insert user into database
    const result = await db.run(
      `INSERT INTO users (
        email, password_hash, first_name, last_name, phone, 
        date_of_birth, gender, blood_type, allergies, 
        insurance_provider, policy_number, group_number
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      email.toLowerCase(),
      passwordHash,
      firstName,
      lastName,
      phone,
      dateOfBirth,
      gender,
      bloodType || null,
      allergies || null,
      insuranceProvider || null,
      policyNumber || null,
      groupNumber || null
    );

    const userId = result.lastID;

    // Insert current medications if provided
    if (currentMedications.length > 0) {
      for (const medication of currentMedications) {
        await db.run(
          `INSERT INTO current_medications (user_id, medication_name) VALUES (?, ?)`,
          userId,
          medication
        );
      }
    }

    // Generate JWT token
    const token = jwt.sign(
      { 
        userId: userId, 
        email: email.toLowerCase(),
        role: 'patient'
      },
      JWT_SECRET,
      { expiresIn: '7d' }
    );

    // Return success response (don't include password hash)
    res.status(201).json({
      message: "User registered successfully",
      user: {
        id: userId,
        email: email.toLowerCase(),
        firstName,
        lastName,
        phone,
        dateOfBirth,
        gender,
        bloodType,
        role: 'patient'
      },
      token
    });

  } catch (error) {
    console.error("Registration error:", error);
    res.status(500).json({
      error: "Internal server error",
      message: "Failed to register user"
    });
  }
};

export const loginUser: RequestHandler = async (req, res) => {
  try {
    const { email, password }: LoginRequest = req.body;

    // Validate required fields
    if (!email || !password) {
      return res.status(400).json({
        error: "Missing required fields",
        details: "Email and password are required"
      });
    }

    const db = await getDatabase();

    // Get user from database
    const user = await db.get(
      "SELECT * FROM users WHERE email = ?",
      email.toLowerCase()
    );

    if (!user) {
      return res.status(401).json({
        error: "Invalid credentials",
        message: "No account found with this email address."
      });
    }

    // Verify password
    const passwordMatch = await bcrypt.compare(password, user.password_hash);

    if (!passwordMatch) {
      return res.status(401).json({
        error: "Invalid credentials",
        message: "Incorrect password."
      });
    }

    // Generate JWT token
    const token = jwt.sign(
      { 
        userId: user.id, 
        email: user.email,
        role: user.role
      },
      JWT_SECRET,
      { expiresIn: '7d' }
    );

    // Return success response (don't include password hash)
    res.json({
      message: "Login successful",
      user: {
        id: user.id,
        email: user.email,
        firstName: user.first_name,
        lastName: user.last_name,
        phone: user.phone,
        dateOfBirth: user.date_of_birth,
        gender: user.gender,
        bloodType: user.blood_type,
        role: user.role
      },
      token
    });

  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({
      error: "Internal server error",
      message: "Failed to login user"
    });
  }
};

export const checkUserExists: RequestHandler = async (req, res) => {
  try {
    const { email } = req.params;

    if (!email) {
      return res.status(400).json({
        error: "Email parameter is required"
      });
    }

    const db = await getDatabase();

    const user = await db.get(
      "SELECT id FROM users WHERE email = ?",
      email.toLowerCase()
    );

    res.json({
      exists: !!user,
      message: user ? "User exists" : "User not found"
    });

  } catch (error) {
    console.error("Check user exists error:", error);
    res.status(500).json({
      error: "Internal server error",
      message: "Failed to check if user exists"
    });
  }
};

// Middleware to verify JWT token
export const authenticateToken: RequestHandler = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (!token) {
    return res.status(401).json({
      error: "Access denied",
      message: "No token provided"
    });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET) as any;
    (req as any).user = decoded;
    next();
  } catch (error) {
    res.status(403).json({
      error: "Invalid token",
      message: "Token is not valid"
    });
  }
};
