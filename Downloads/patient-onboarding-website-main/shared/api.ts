/**
 * Shared code between client and server
 * Useful to share types between client and server
 * and/or small pure JS functions that can be used on both client and server
 */

/**
 * Example response type for /api/demo
 */
export interface DemoResponse {
  message: string;
}

// Authentication interfaces
export interface RegisterRequest {
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

export interface LoginRequest {
  email: string;
  password: string;
}

export interface AuthResponse {
  message: string;
  user: {
    id: number;
    email: string;
    firstName: string;
    lastName: string;
    phone: string;
    dateOfBirth: string;
    gender: string;
    bloodType?: string;
    role: string;
  };
  token: string;
}

export interface CheckUserResponse {
  exists: boolean;
  message: string;
}

// Medical data interfaces
export interface EmergencyContact {
  id?: string;
  firstName: string;
  lastName: string;
  relationship: string;
  primaryPhone: string;
  secondaryPhone?: string;
  email?: string;
  address: {
    street: string;
    city: string;
    state: string;
    zipCode: string;
  };
  isAuthorizedToReceiveInfo: boolean;
  canMakeHealthcareDecisions: boolean;
  notes?: string;
}

export interface MedicalCondition {
  condition: string;
  diagnosedYear: string;
  status: "current" | "past" | "family-history";
  notes?: string;
}

export interface Medication {
  name: string;
  dosage: string;
  frequency: string;
  prescribedBy: string;
  startDate: string;
  notes?: string;
}

export interface Surgery {
  procedure: string;
  date: string;
  hospital: string;
  surgeon: string;
  complications?: string;
}

export interface FamilyHistory {
  relation: string;
  conditions: string[];
  ageAtDeath?: string;
  causeOfDeath?: string;
}

export interface MedicalHistoryData {
  currentConditions: MedicalCondition[];
  pastConditions: MedicalCondition[];
  currentMedications: Medication[];
  allergies: {
    medications: string[];
    foods: string[];
    environmental: string[];
    other: string;
  };
  surgeries: Surgery[];
  familyHistory: FamilyHistory[];
  lifestyle: {
    smoking: string;
    smokingDetails?: string;
    alcohol: string;
    alcoholDetails?: string;
    exercise: string;
    exerciseDetails?: string;
    diet: string;
    dietDetails?: string;
  };
  additionalInfo: {
    hospitalizations: string;
    emergencyRoomVisits: string;
    significantInjuries: string;
    otherConcerns: string;
  };
}

export interface AppointmentRequest {
  date: string;
  time: string;
  doctorName: string;
  appointmentType: string;
  location: string;
  notes?: string;
}

export interface AppointmentResponse {
  id: string;
  date: string;
  time: string;
  doctorName: string;
  appointmentType: string;
  status: "scheduled" | "completed" | "cancelled";
  location: string;
  notes?: string;
}
