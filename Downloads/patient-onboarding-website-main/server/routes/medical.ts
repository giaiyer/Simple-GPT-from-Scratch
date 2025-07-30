import { RequestHandler } from "express";
import { getDatabase } from "../database";

interface EmergencyContactRequest {
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

interface MedicalHistoryRequest {
  currentConditions: Array<{
    condition: string;
    diagnosedYear: string;
    notes?: string;
  }>;
  pastConditions: Array<{
    condition: string;
    diagnosedYear: string;
    notes?: string;
  }>;
  currentMedications: Array<{
    name: string;
    dosage: string;
    frequency: string;
    prescribedBy: string;
    startDate: string;
    notes?: string;
  }>;
  allergies: {
    medications: string[];
    foods: string[];
    environmental: string[];
    other: string;
  };
  surgeries: Array<{
    procedure: string;
    date: string;
    hospital: string;
    surgeon: string;
    complications?: string;
  }>;
  familyHistory: Array<{
    relation: string;
    conditions: string[];
    ageAtDeath?: string;
    causeOfDeath?: string;
  }>;
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

interface AppointmentRequest {
  date: string;
  time: string;
  doctorName: string;
  appointmentType: string;
  location: string;
  notes?: string;
}

// Emergency Contacts Routes
export const saveEmergencyContacts: RequestHandler = async (req, res) => {
  try {
    const userId = (req as any).user.userId;
    const contacts: EmergencyContactRequest[] = req.body.contacts;

    if (!contacts || !Array.isArray(contacts)) {
      return res.status(400).json({
        error: "Invalid request",
        message: "Contacts array is required"
      });
    }

    const db = await getDatabase();

    // Delete existing emergency contacts for this user
    await db.run("DELETE FROM emergency_contacts WHERE user_id = ?", userId);

    // Insert new emergency contacts
    for (const contact of contacts) {
      await db.run(
        `INSERT INTO emergency_contacts (
          user_id, first_name, last_name, relationship, primary_phone,
          secondary_phone, email, street_address, city, state, zip_code,
          is_authorized_to_receive_info, can_make_healthcare_decisions, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        userId,
        contact.firstName,
        contact.lastName,
        contact.relationship,
        contact.primaryPhone,
        contact.secondaryPhone || null,
        contact.email || null,
        contact.address.street,
        contact.address.city,
        contact.address.state,
        contact.address.zipCode,
        contact.isAuthorizedToReceiveInfo,
        contact.canMakeHealthcareDecisions,
        contact.notes || null
      );
    }

    res.json({
      message: "Emergency contacts saved successfully",
      count: contacts.length
    });

  } catch (error) {
    console.error("Save emergency contacts error:", error);
    res.status(500).json({
      error: "Internal server error",
      message: "Failed to save emergency contacts"
    });
  }
};

export const getEmergencyContacts: RequestHandler = async (req, res) => {
  try {
    const userId = (req as any).user.userId;
    const db = await getDatabase();

    const contacts = await db.all(
      "SELECT * FROM emergency_contacts WHERE user_id = ? ORDER BY created_at ASC",
      userId
    );

    // Transform database results to match frontend format
    const formattedContacts = contacts.map(contact => ({
      id: contact.id.toString(),
      firstName: contact.first_name,
      lastName: contact.last_name,
      relationship: contact.relationship,
      primaryPhone: contact.primary_phone,
      secondaryPhone: contact.secondary_phone,
      email: contact.email,
      address: {
        street: contact.street_address,
        city: contact.city,
        state: contact.state,
        zipCode: contact.zip_code
      },
      isAuthorizedToReceiveInfo: !!contact.is_authorized_to_receive_info,
      canMakeHealthcareDecisions: !!contact.can_make_healthcare_decisions,
      notes: contact.notes
    }));

    res.json({
      contacts: formattedContacts
    });

  } catch (error) {
    console.error("Get emergency contacts error:", error);
    res.status(500).json({
      error: "Internal server error",
      message: "Failed to get emergency contacts"
    });
  }
};

// Medical History Routes
export const saveMedicalHistory: RequestHandler = async (req, res) => {
  try {
    const userId = (req as any).user.userId;
    const data: MedicalHistoryRequest = req.body;

    const db = await getDatabase();

    // Clear existing medical history data for this user
    await db.run("DELETE FROM medical_history WHERE user_id = ?", userId);
    await db.run("DELETE FROM current_medications WHERE user_id = ?", userId);
    await db.run("DELETE FROM surgery_history WHERE user_id = ?", userId);
    await db.run("DELETE FROM family_history WHERE user_id = ?", userId);
    await db.run("DELETE FROM lifestyle_info WHERE user_id = ?", userId);
    await db.run("DELETE FROM additional_health_info WHERE user_id = ?", userId);

    // Save current conditions
    for (const condition of data.currentConditions) {
      await db.run(
        `INSERT INTO medical_history (user_id, condition_name, diagnosed_year, status, notes)
         VALUES (?, ?, ?, ?, ?)`,
        userId,
        condition.condition,
        condition.diagnosedYear,
        'current',
        condition.notes || null
      );
    }

    // Save past conditions
    for (const condition of data.pastConditions) {
      await db.run(
        `INSERT INTO medical_history (user_id, condition_name, diagnosed_year, status, notes)
         VALUES (?, ?, ?, ?, ?)`,
        userId,
        condition.condition,
        condition.diagnosedYear,
        'past',
        condition.notes || null
      );
    }

    // Save current medications
    for (const medication of data.currentMedications) {
      await db.run(
        `INSERT INTO current_medications (
          user_id, medication_name, dosage, frequency, prescribed_by, start_date, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
        userId,
        medication.name,
        medication.dosage,
        medication.frequency,
        medication.prescribedBy,
        medication.startDate,
        medication.notes || null
      );
    }

    // Save surgeries
    for (const surgery of data.surgeries) {
      await db.run(
        `INSERT INTO surgery_history (
          user_id, procedure_name, surgery_date, hospital, surgeon, complications
        ) VALUES (?, ?, ?, ?, ?, ?)`,
        userId,
        surgery.procedure,
        surgery.date,
        surgery.hospital,
        surgery.surgeon,
        surgery.complications || null
      );
    }

    // Save family history
    for (const family of data.familyHistory) {
      await db.run(
        `INSERT INTO family_history (
          user_id, relation, medical_conditions, age_at_death, cause_of_death
        ) VALUES (?, ?, ?, ?, ?)`,
        userId,
        family.relation,
        JSON.stringify(family.conditions),
        family.ageAtDeath || null,
        family.causeOfDeath || null
      );
    }

    // Save lifestyle information
    await db.run(
      `INSERT INTO lifestyle_info (
        user_id, smoking_status, smoking_details, alcohol_consumption, alcohol_details,
        exercise_frequency, exercise_details, diet_type, diet_details
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      userId,
      data.lifestyle.smoking,
      data.lifestyle.smokingDetails || null,
      data.lifestyle.alcohol,
      data.lifestyle.alcoholDetails || null,
      data.lifestyle.exercise,
      data.lifestyle.exerciseDetails || null,
      data.lifestyle.diet,
      data.lifestyle.dietDetails || null
    );

    // Save additional health info
    await db.run(
      `INSERT INTO additional_health_info (
        user_id, hospitalizations, emergency_room_visits, significant_injuries, other_concerns
      ) VALUES (?, ?, ?, ?, ?)`,
      userId,
      data.additionalInfo.hospitalizations,
      data.additionalInfo.emergencyRoomVisits,
      data.additionalInfo.significantInjuries,
      data.additionalInfo.otherConcerns
    );

    // Update user allergies
    const allergiesText = [
      ...data.allergies.medications.map(a => `Medication: ${a}`),
      ...data.allergies.foods.map(a => `Food: ${a}`),
      ...data.allergies.environmental.map(a => `Environmental: ${a}`),
      data.allergies.other ? `Other: ${data.allergies.other}` : ''
    ].filter(Boolean).join('; ');

    await db.run(
      "UPDATE users SET allergies = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
      allergiesText || null,
      userId
    );

    res.json({
      message: "Medical history saved successfully"
    });

  } catch (error) {
    console.error("Save medical history error:", error);
    res.status(500).json({
      error: "Internal server error",
      message: "Failed to save medical history"
    });
  }
};

export const getMedicalHistory: RequestHandler = async (req, res) => {
  try {
    const userId = (req as any).user.userId;
    const db = await getDatabase();

    // Get all medical history data
    const [
      conditions,
      medications,
      surgeries,
      familyHistory,
      lifestyle,
      additionalInfo,
      user
    ] = await Promise.all([
      db.all("SELECT * FROM medical_history WHERE user_id = ?", userId),
      db.all("SELECT * FROM current_medications WHERE user_id = ?", userId),
      db.all("SELECT * FROM surgery_history WHERE user_id = ?", userId),
      db.all("SELECT * FROM family_history WHERE user_id = ?", userId),
      db.get("SELECT * FROM lifestyle_info WHERE user_id = ?", userId),
      db.get("SELECT * FROM additional_health_info WHERE user_id = ?", userId),
      db.get("SELECT allergies FROM users WHERE id = ?", userId)
    ]);

    // Parse allergies
    const allergies = {
      medications: [],
      foods: [],
      environmental: [],
      other: ''
    };

    if (user?.allergies) {
      const allergyItems = user.allergies.split(';').map((item: string) => item.trim());
      for (const item of allergyItems) {
        if (item.startsWith('Medication:')) {
          allergies.medications.push(item.replace('Medication:', '').trim());
        } else if (item.startsWith('Food:')) {
          allergies.foods.push(item.replace('Food:', '').trim());
        } else if (item.startsWith('Environmental:')) {
          allergies.environmental.push(item.replace('Environmental:', '').trim());
        } else if (item.startsWith('Other:')) {
          allergies.other = item.replace('Other:', '').trim();
        }
      }
    }

    // Format response
    const medicalHistoryData = {
      currentConditions: conditions
        .filter(c => c.status === 'current')
        .map(c => ({
          condition: c.condition_name,
          diagnosedYear: c.diagnosed_year,
          status: c.status,
          notes: c.notes
        })),
      pastConditions: conditions
        .filter(c => c.status === 'past')
        .map(c => ({
          condition: c.condition_name,
          diagnosedYear: c.diagnosed_year,
          status: c.status,
          notes: c.notes
        })),
      currentMedications: medications.map(m => ({
        name: m.medication_name,
        dosage: m.dosage,
        frequency: m.frequency,
        prescribedBy: m.prescribed_by,
        startDate: m.start_date,
        notes: m.notes
      })),
      allergies,
      surgeries: surgeries.map(s => ({
        procedure: s.procedure_name,
        date: s.surgery_date,
        hospital: s.hospital,
        surgeon: s.surgeon,
        complications: s.complications
      })),
      familyHistory: familyHistory.map(f => ({
        relation: f.relation,
        conditions: JSON.parse(f.medical_conditions || '[]'),
        ageAtDeath: f.age_at_death,
        causeOfDeath: f.cause_of_death
      })),
      lifestyle: lifestyle ? {
        smoking: lifestyle.smoking_status,
        smokingDetails: lifestyle.smoking_details,
        alcohol: lifestyle.alcohol_consumption,
        alcoholDetails: lifestyle.alcohol_details,
        exercise: lifestyle.exercise_frequency,
        exerciseDetails: lifestyle.exercise_details,
        diet: lifestyle.diet_type,
        dietDetails: lifestyle.diet_details
      } : {
        smoking: '',
        alcohol: '',
        exercise: '',
        diet: ''
      },
      additionalInfo: additionalInfo ? {
        hospitalizations: additionalInfo.hospitalizations,
        emergencyRoomVisits: additionalInfo.emergency_room_visits,
        significantInjuries: additionalInfo.significant_injuries,
        otherConcerns: additionalInfo.other_concerns
      } : {
        hospitalizations: '',
        emergencyRoomVisits: '',
        significantInjuries: '',
        otherConcerns: ''
      }
    };

    res.json(medicalHistoryData);

  } catch (error) {
    console.error("Get medical history error:", error);
    res.status(500).json({
      error: "Internal server error",
      message: "Failed to get medical history"
    });
  }
};

// Appointments Routes
export const createAppointment: RequestHandler = async (req, res) => {
  try {
    const userId = (req as any).user.userId;
    const { date, time, doctorName, appointmentType, location, notes }: AppointmentRequest = req.body;

    if (!date || !time || !doctorName || !appointmentType || !location) {
      return res.status(400).json({
        error: "Missing required fields",
        details: "date, time, doctorName, appointmentType, and location are required"
      });
    }

    const db = await getDatabase();

    const result = await db.run(
      `INSERT INTO appointments (
        user_id, appointment_date, appointment_time, doctor_name, 
        appointment_type, location, notes
      ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      userId,
      date,
      time,
      doctorName,
      appointmentType,
      location,
      notes || null
    );

    res.status(201).json({
      message: "Appointment created successfully",
      appointment: {
        id: result.lastID,
        date,
        time,
        doctorName,
        appointmentType,
        location,
        status: 'scheduled',
        notes
      }
    });

  } catch (error) {
    console.error("Create appointment error:", error);
    res.status(500).json({
      error: "Internal server error",
      message: "Failed to create appointment"
    });
  }
};

export const getAppointments: RequestHandler = async (req, res) => {
  try {
    const userId = (req as any).user.userId;
    const db = await getDatabase();

    const appointments = await db.all(
      "SELECT * FROM appointments WHERE user_id = ? ORDER BY appointment_date DESC, appointment_time DESC",
      userId
    );

    const formattedAppointments = appointments.map(apt => ({
      id: apt.id.toString(),
      date: apt.appointment_date,
      time: apt.appointment_time,
      doctorName: apt.doctor_name,
      appointmentType: apt.appointment_type,
      status: apt.status,
      location: apt.location,
      notes: apt.notes
    }));

    res.json({
      appointments: formattedAppointments
    });

  } catch (error) {
    console.error("Get appointments error:", error);
    res.status(500).json({
      error: "Internal server error",
      message: "Failed to get appointments"
    });
  }
};

export const updateAppointment: RequestHandler = async (req, res) => {
  try {
    const userId = (req as any).user.userId;
    const appointmentId = req.params.id;
    const updates = req.body;

    const db = await getDatabase();

    // Verify appointment belongs to user
    const appointment = await db.get(
      "SELECT id FROM appointments WHERE id = ? AND user_id = ?",
      appointmentId,
      userId
    );

    if (!appointment) {
      return res.status(404).json({
        error: "Appointment not found"
      });
    }

    // Build update query
    const fields = [];
    const values = [];

    if (updates.date) {
      fields.push("appointment_date = ?");
      values.push(updates.date);
    }
    if (updates.time) {
      fields.push("appointment_time = ?");
      values.push(updates.time);
    }
    if (updates.doctorName) {
      fields.push("doctor_name = ?");
      values.push(updates.doctorName);
    }
    if (updates.appointmentType) {
      fields.push("appointment_type = ?");
      values.push(updates.appointmentType);
    }
    if (updates.location) {
      fields.push("location = ?");
      values.push(updates.location);
    }
    if (updates.status) {
      fields.push("status = ?");
      values.push(updates.status);
    }
    if (updates.notes !== undefined) {
      fields.push("notes = ?");
      values.push(updates.notes);
    }

    if (fields.length === 0) {
      return res.status(400).json({
        error: "No fields to update"
      });
    }

    fields.push("updated_at = CURRENT_TIMESTAMP");
    values.push(appointmentId);

    await db.run(
      `UPDATE appointments SET ${fields.join(", ")} WHERE id = ?`,
      ...values
    );

    res.json({
      message: "Appointment updated successfully"
    });

  } catch (error) {
    console.error("Update appointment error:", error);
    res.status(500).json({
      error: "Internal server error",
      message: "Failed to update appointment"
    });
  }
};
