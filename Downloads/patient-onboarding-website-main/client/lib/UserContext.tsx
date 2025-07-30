import React, { createContext, useContext, useState, useEffect } from "react";

export interface MedicalCondition {
  condition: string;
  diagnosedYear: string;
  status: "current" | "past" | "family-history";
  notes?: string;
}

export interface EmergencyContact {
  id: string;
  firstName: string;
  lastName: string;
  relationship: string;
  primaryPhone: string;
  secondaryPhone?: string;
  email?: string;
}

export interface UserData {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  dateOfBirth: string;
  gender: string;
  role: "patient" | "admin";
  medicalConditions?: MedicalCondition[];
  emergencyContacts?: EmergencyContact[];
}

export interface OnboardingProgress {
  uploadDocuments: boolean;
  scheduleAppointment: boolean;
  consentForms: boolean;
  medicalHistory: boolean;
  emergencyContacts: boolean;
}

export interface HealthcareTasksState {
  documentVerificationStatus: "pending" | "in_progress" | "completed";
  callScheduled: boolean;
  selectedTimeSlot: "morning" | "afternoon" | "evening" | "anytime" | null;
  notificationsConfigured: boolean;
  selectedDeliveryMethods: ("email" | "sms")[];
  appointmentReminders: boolean;
  testResults: boolean;
  healthTips: boolean;
  wellnessEnrolled: boolean;
  selectedWellnessPlan: "basic" | "premium" | null;
}

export interface AppointmentData {
  id: string;
  date: string;
  time: string;
  doctorName: string;
  appointmentType: string;
  status: "scheduled" | "completed" | "cancelled";
  location: string;
}

interface UserProfile {
  userData: UserData;
  onboardingProgress: OnboardingProgress;
  appointments: AppointmentData[];
  healthcareTasks: HealthcareTasksState;
}

interface UserContextType {
  userData: UserData | null;
  currentUserEmail: string | null;
  setUserData: (data: UserData) => void;
  onboardingProgress: OnboardingProgress;
  updateOnboardingProgress: (
    task: keyof OnboardingProgress,
    completed: boolean,
  ) => void;
  appointments: AppointmentData[];
  addAppointment: (appointment: AppointmentData) => void;
  loadAppointments: (appointments: AppointmentData[]) => void;
  updateAppointment: (
    appointmentId: string,
    updatedAppointment: Partial<AppointmentData>,
  ) => void;
  getCompletionPercentage: () => number;
  getCompletedTasksCount: () => number;
  loginUser: (email: string) => boolean;
  loginAdmin: (email: string, password: string) => boolean;
  logoutUser: () => void;
  userExists: (email: string) => Promise<boolean>;
  resetOnboarding: () => void;
  isAdmin: () => boolean;
  healthcareTasks: HealthcareTasksState;
  updateHealthcareTask: <K extends keyof HealthcareTasksState>(
    task: K,
    value: HealthcareTasksState[K],
  ) => void;
  updateHealthcareTasksBatch: (updates: Partial<HealthcareTasksState>) => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

const initialProgress: OnboardingProgress = {
  uploadDocuments: false,
  scheduleAppointment: false,
  consentForms: false,
  medicalHistory: false,
  emergencyContacts: false,
};

const initialHealthcareTasks: HealthcareTasksState = {
  documentVerificationStatus: "pending",
  callScheduled: false,
  selectedTimeSlot: null,
  notificationsConfigured: false,
  selectedDeliveryMethods: ["email"],
  appointmentReminders: true,
  testResults: true,
  healthTips: true,
  wellnessEnrolled: false,
  selectedWellnessPlan: null,
};

export function UserProvider({ children }: { children: React.ReactNode }) {
  const [currentUserEmail, setCurrentUserEmail] = useState<string | null>(null);
  const [userData, setUserDataState] = useState<UserData | null>(null);
  const [onboardingProgress, setOnboardingProgress] =
    useState<OnboardingProgress>(initialProgress);
  const [appointments, setAppointments] = useState<AppointmentData[]>([]);
  const [healthcareTasks, setHealthcareTasks] = useState<HealthcareTasksState>(
    initialHealthcareTasks,
  );

  // Load current user on mount
  useEffect(() => {
    const token = localStorage.getItem("healthcarePlus_token");
    const savedCurrentUser = localStorage.getItem("healthcarePlus_currentUser");

    if (token && savedCurrentUser) {
      // For now, we'll continue to use localStorage for profile data
      // In a full implementation, you'd want to validate the token with the server
      const success = loginUser(savedCurrentUser);
      if (!success) {
        localStorage.removeItem("healthcarePlus_currentUser");
        localStorage.removeItem("healthcarePlus_token");
      }
    }
  }, []);

  const getUserKey = (email: string) => `healthcarePlus_user_${email}`;

  const loadUserProfile = (email: string): UserProfile | null => {
    const userKey = getUserKey(email);
    const savedProfile = localStorage.getItem(userKey);
    if (savedProfile) {
      try {
        return JSON.parse(savedProfile);
      } catch (error) {
        console.error("Error loading user profile:", error);
      }
    }
    return null;
  };

  const saveUserProfile = (email: string, profile: UserProfile) => {
    const userKey = getUserKey(email);
    localStorage.setItem(userKey, JSON.stringify(profile));
  };

  const userExists = async (email: string): Promise<boolean> => {
    try {
      const response = await fetch(`/api/auth/check-user/${encodeURIComponent(email)}`);
      const data = await response.json();
      return data.exists;
    } catch (error) {
      console.error("Error checking if user exists:", error);
      return false;
    }
  };

  const loginUser = (email: string): boolean => {
    // This method is now primarily used for setting user data after successful API login
    setCurrentUserEmail(email);

    // Load existing onboarding progress and other data from localStorage if available
    const existingProfile = loadUserProfile(email);
    if (existingProfile) {
      setOnboardingProgress(existingProfile.onboardingProgress);
      setAppointments(existingProfile.appointments || []);
      setHealthcareTasks(existingProfile.healthcareTasks || initialHealthcareTasks);
    }

    localStorage.setItem("healthcarePlus_currentUser", email);
    return true;
  };

  // Admin login with predefined credentials
  const loginAdmin = (email: string, password: string): boolean => {
    // Single admin credential
    if (email === "admin@healthcare.com" && password === "admin123") {
      const adminUserData: UserData = {
        firstName: "Healthcare",
        lastName: "Admin",
        email: "admin@healthcare.com",
        phone: "+91 98765 43210",
        dateOfBirth: "1980-01-01",
        gender: "other",
        role: "admin",
      };

      setCurrentUserEmail(email);
      setUserDataState(adminUserData);
      setOnboardingProgress(initialProgress);
      setAppointments([]);
      setHealthcareTasks(initialHealthcareTasks);
      localStorage.setItem("healthcarePlus_currentUser", email);
      localStorage.setItem("healthcarePlus_userRole", "admin");
      return true;
    }
    return false;
  };

  const logoutUser = () => {
    setCurrentUserEmail(null);
    setUserDataState(null);
    setOnboardingProgress(initialProgress);
    setAppointments([]);
    setHealthcareTasks(initialHealthcareTasks);
    localStorage.removeItem("healthcarePlus_currentUser");
    localStorage.removeItem("healthcarePlus_userRole");
  };

  const isAdmin = (): boolean => {
    return (
      userData?.role === "admin" ||
      localStorage.getItem("healthcarePlus_userRole") === "admin"
    );
  };

  const setUserData = (data: UserData) => {
    setUserDataState(data);
    if (currentUserEmail || data.email) {
      const email = currentUserEmail || data.email;
      const profile: UserProfile = {
        userData: data,
        onboardingProgress,
        appointments,
        healthcareTasks,
      };
      saveUserProfile(email, profile);
      setCurrentUserEmail(email);
      localStorage.setItem("healthcarePlus_currentUser", email);
    }
  };

  const updateOnboardingProgress = (
    task: keyof OnboardingProgress,
    completed: boolean,
  ) => {
    const newProgress = { ...onboardingProgress, [task]: completed };
    setOnboardingProgress(newProgress);

    if (currentUserEmail && userData) {
      const profile: UserProfile = {
        userData,
        onboardingProgress: newProgress,
        appointments,
        healthcareTasks,
      };
      saveUserProfile(currentUserEmail, profile);
    }
  };

  const addAppointment = (appointment: AppointmentData) => {
    const newAppointments = [...appointments, appointment];
    setAppointments(newAppointments);

    if (currentUserEmail && userData) {
      const profile: UserProfile = {
        userData,
        onboardingProgress,
        appointments: newAppointments,
        healthcareTasks,
      };
      saveUserProfile(currentUserEmail, profile);
    }
  };

  const loadAppointments = (appointmentsList: AppointmentData[]) => {
    setAppointments(appointmentsList);

    if (currentUserEmail && userData) {
      const profile: UserProfile = {
        userData,
        onboardingProgress,
        appointments: appointmentsList,
        healthcareTasks,
      };
      saveUserProfile(currentUserEmail, profile);
    }
  };

  const updateAppointment = (
    appointmentId: string,
    updatedAppointment: Partial<AppointmentData>,
  ) => {
    const newAppointments = appointments.map((apt) =>
      apt.id === appointmentId ? { ...apt, ...updatedAppointment } : apt,
    );
    setAppointments(newAppointments);

    if (currentUserEmail && userData) {
      const profile: UserProfile = {
        userData,
        onboardingProgress,
        appointments: newAppointments,
        healthcareTasks,
      };
      saveUserProfile(currentUserEmail, profile);
    }
  };

  const getCompletionPercentage = (): number => {
    const tasks = Object.values(onboardingProgress);
    const completedTasks = tasks.filter(Boolean).length;
    return Math.round((completedTasks / tasks.length) * 100);
  };

  const getCompletedTasksCount = (): number => {
    return Object.values(onboardingProgress).filter(Boolean).length;
  };

  const updateHealthcareTask = <K extends keyof HealthcareTasksState>(
    task: K,
    value: HealthcareTasksState[K],
  ) => {
    const newTasks = { ...healthcareTasks, [task]: value };
    setHealthcareTasks(newTasks);

    if (currentUserEmail && userData) {
      const profile: UserProfile = {
        userData,
        onboardingProgress,
        appointments,
        healthcareTasks: newTasks,
      };
      saveUserProfile(currentUserEmail, profile);
    }
  };

  const updateHealthcareTasksBatch = (
    updates: Partial<HealthcareTasksState>,
  ) => {
    const newTasks = { ...healthcareTasks, ...updates };
    setHealthcareTasks(newTasks);

    if (currentUserEmail && userData) {
      const profile: UserProfile = {
        userData,
        onboardingProgress,
        appointments,
        healthcareTasks: newTasks,
      };
      saveUserProfile(currentUserEmail, profile);
    }
  };

  const resetOnboarding = () => {
    setOnboardingProgress(initialProgress);
    if (currentUserEmail && userData) {
      const profile: UserProfile = {
        userData,
        onboardingProgress: initialProgress,
        appointments,
        healthcareTasks,
      };
      saveUserProfile(currentUserEmail, profile);
    }
  };

  return (
    <UserContext.Provider
      value={{
        userData,
        currentUserEmail,
        setUserData,
        onboardingProgress,
        updateOnboardingProgress,
        appointments,
        addAppointment,
        loadAppointments,
        updateAppointment,
        getCompletionPercentage,
        getCompletedTasksCount,
        loginUser,
        loginAdmin,
        logoutUser,
        userExists,
        resetOnboarding,
        isAdmin,
        healthcareTasks,
        updateHealthcareTask,
        updateHealthcareTasksBatch,
      }}
    >
      {children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error("useUser must be used within a UserProvider");
  }
  return context;
}
