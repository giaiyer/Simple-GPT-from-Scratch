import { RequestHandler } from "express";

export interface NotificationPreferences {
  appointmentReminders: boolean;
  testResults: boolean;
  healthTips: boolean;
  deliveryMethods: ("email" | "sms")[];
  phoneNumber?: string;
  email?: string;
}

// Simple in-memory storage for demo purposes
// In production, this would be saved to a database
const userPreferences: Map<string, NotificationPreferences> = new Map();

export const updateNotificationPreferences: RequestHandler = async (
  req,
  res,
) => {
  try {
    const preferences: NotificationPreferences = req.body;

    // Validate preferences
    if (
      !preferences.deliveryMethods ||
      preferences.deliveryMethods.length === 0
    ) {
      return res.status(400).json({
        success: false,
        error: "At least one delivery method must be selected",
      });
    }

    if (
      preferences.deliveryMethods.includes("sms") &&
      !preferences.phoneNumber
    ) {
      return res.status(400).json({
        success: false,
        error: "Phone number required for SMS notifications",
      });
    }

    if (preferences.deliveryMethods.includes("email") && !preferences.email) {
      return res.status(400).json({
        success: false,
        error: "Email address required for email notifications",
      });
    }

    // Save preferences (in production: save to database)
    const userKey = preferences.email || preferences.phoneNumber || "default";
    userPreferences.set(userKey, preferences);

    console.log("💾 Notification preferences saved:", preferences);

    res.json({
      success: true,
      message: "Notification preferences saved successfully",
      preferences: preferences,
    });
  } catch (error) {
    console.error("Error saving notification preferences:", error);
    res.status(500).json({
      success: false,
      error: "Failed to save notification preferences",
    });
  }
};

export const getNotificationPreferences: RequestHandler = async (req, res) => {
  try {
    const { email, phoneNumber } = req.query;
    const userKey = (email as string) || (phoneNumber as string) || "default";

    const preferences = userPreferences.get(userKey);

    if (preferences) {
      res.json({
        success: true,
        preferences,
      });
    } else {
      res.json({
        success: true,
        preferences: {
          appointmentReminders: true,
          testResults: true,
          healthTips: true,
          deliveryMethods: ["email"],
          phoneNumber: phoneNumber as string,
          email: email as string,
        },
      });
    }
  } catch (error) {
    console.error("Error getting notification preferences:", error);
    res.status(500).json({
      success: false,
      error: "Failed to get notification preferences",
    });
  }
};

export const testNotificationSettings: RequestHandler = async (req, res) => {
  try {
    const { method, recipient } = req.body;

    if (!method || !recipient) {
      return res.status(400).json({
        success: false,
        error: "Missing method or recipient information",
      });
    }

    // Simulate successful test (no actual sending)
    console.log(`🧪 Test notification settings for ${method}:`, recipient);

    res.json({
      success: true,
      method: method,
      message: `${method === "email" ? "Email" : "SMS"} settings tested successfully`,
      note: "Settings saved - no actual message sent",
    });
  } catch (error) {
    console.error("Error testing notification settings:", error);
    res.status(500).json({
      success: false,
      error: "Failed to test notification settings",
    });
  }
};
