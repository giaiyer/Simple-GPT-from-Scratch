import express from "express";
import cors from "cors";
import { handleDemo } from "./routes/demo";
import {
  updateNotificationPreferences,
  getNotificationPreferences,
  testNotificationSettings,
} from "./routes/simpleNotifications";
import {
  registerUser,
  loginUser,
  checkUserExists,
  authenticateToken
} from "./routes/auth";
import {
  saveEmergencyContacts,
  getEmergencyContacts,
  saveMedicalHistory,
  getMedicalHistory,
  createAppointment,
  getAppointments,
  updateAppointment
} from "./routes/medical";
import { getDatabase } from "./database";

export function createServer() {
  const app = express();

  // Middleware
  app.use(cors());
  app.use(express.json({ limit: '10mb' }));
  app.use(express.urlencoded({ extended: true, limit: '10mb' }));

  // Initialize database on server start
  app.use(async (_req, _res, next) => {
    try {
      await getDatabase();
      next();
    } catch (error) {
      console.error("Database initialization error:", error);
      next();
    }
  });

  // Example API routes
  app.get("/api/ping", (_req, res) => {
    res.json({ message: "Hello from Express server v2!" });
  });

  app.get("/api/demo", handleDemo);

  // Authentication routes
  app.post("/api/auth/register", registerUser);
  app.post("/api/auth/login", loginUser);
  app.get("/api/auth/check-user/:email", checkUserExists);

  // Protected medical data routes (require authentication)
  app.post("/api/medical/emergency-contacts", authenticateToken, saveEmergencyContacts);
  app.get("/api/medical/emergency-contacts", authenticateToken, getEmergencyContacts);
  app.post("/api/medical/history", authenticateToken, saveMedicalHistory);
  app.get("/api/medical/history", authenticateToken, getMedicalHistory);
  app.post("/api/medical/appointments", authenticateToken, createAppointment);
  app.get("/api/medical/appointments", authenticateToken, getAppointments);
  app.put("/api/medical/appointments/:id", authenticateToken, updateAppointment);

  // Simple notification preference routes (no actual sending)
  app.post("/api/notifications/preferences", updateNotificationPreferences);
  app.get("/api/notifications/preferences", getNotificationPreferences);
  app.post("/api/notifications/test", testNotificationSettings);

  return app;
}
