import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "@/lib/UserContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Heart,
  ArrowLeft,
  Stethoscope,
  Activity,
  Pill,
  Users,
  AlertTriangle,
  CheckCircle2,
  Plus,
  X,
  Calendar,
} from "lucide-react";

interface MedicalCondition {
  condition: string;
  diagnosedYear: string;
  status: "current" | "past" | "family-history";
  notes?: string;
}

interface Medication {
  name: string;
  dosage: string;
  frequency: string;
  prescribedBy: string;
  startDate: string;
  notes?: string;
}

interface Surgery {
  procedure: string;
  date: string;
  hospital: string;
  surgeon: string;
  complications?: string;
}

interface FamilyHistory {
  relation: string;
  conditions: string[];
  ageAtDeath?: string;
  causeOfDeath?: string;
}

interface MedicalHistoryData {
  // Personal Medical History
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

  // Family History
  familyHistory: FamilyHistory[];

  // Lifestyle
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

  // Women's Health
  womensHealth?: {
    lastMenstrualPeriod?: string;
    pregnancies?: string;
    pregnancyComplications?: string;
    contraception?: string;
    mammogramDate?: string;
    papSmearDate?: string;
  };

  // Additional Information
  additionalInfo: {
    hospitalizations: string;
    emergencyRoomVisits: string;
    significantInjuries: string;
    otherConcerns: string;
  };
}

export default function MedicalHistory() {
  const navigate = useNavigate();
  const { updateOnboardingProgress } = useUser();
  const [currentStep, setCurrentStep] = useState(1);
  const [showCompletion, setShowCompletion] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const totalSteps = 5;
  const progressPercentage = (currentStep / totalSteps) * 100;

  const stepTitles = [
    "Current Health",
    "Medical History",
    "Family History",
    "Lifestyle",
    "Additional Info",
  ];

  const [formData, setFormData] = useState<MedicalHistoryData>({
    currentConditions: [],
    pastConditions: [],
    currentMedications: [],
    allergies: {
      medications: [],
      foods: [],
      environmental: [],
      other: "",
    },
    surgeries: [],
    familyHistory: [],
    lifestyle: {
      smoking: "",
      alcohol: "",
      exercise: "",
      diet: "",
    },
    additionalInfo: {
      hospitalizations: "",
      emergencyRoomVisits: "",
      significantInjuries: "",
      otherConcerns: "",
    },
  });

  const commonConditions = [
    "Diabetes",
    "High Blood Pressure",
    "Heart Disease",
    "Asthma",
    "Depression",
    "Anxiety",
    "Arthritis",
    "Cancer",
    "Kidney Disease",
    "Liver Disease",
    "Stroke",
    "High Cholesterol",
  ];

  const familyRelations = [
    "Mother",
    "Father",
    "Sister",
    "Brother",
    "Maternal Grandmother",
    "Maternal Grandfather",
    "Paternal Grandmother",
    "Paternal Grandfather",
    "Aunt",
    "Uncle",
    "Other",
  ];

  const addCondition = (type: "current" | "past", condition: string) => {
    const newCondition: MedicalCondition = {
      condition,
      diagnosedYear: "",
      status: type === "current" ? "current" : "past",
    };

    setFormData((prev) => ({
      ...prev,
      [type === "current" ? "currentConditions" : "pastConditions"]: [
        ...prev[type === "current" ? "currentConditions" : "pastConditions"],
        newCondition,
      ],
    }));
  };

  const removeCondition = (type: "current" | "past", index: number): void => {
    setFormData((prev) => ({
      ...prev,
      [type === "current" ? "currentConditions" : "pastConditions"]: prev[
        type === "current" ? "currentConditions" : "pastConditions"
      ].filter((_, i) => i !== index),
    }));
  };

  const addMedication = () => {
    const newMedication: Medication = {
      name: "",
      dosage: "",
      frequency: "",
      prescribedBy: "",
      startDate: "",
    };

    setFormData((prev) => ({
      ...prev,
      currentMedications: [...prev.currentMedications, newMedication],
    }));
  };

  const removeMedication = (index: number) => {
    setFormData((prev) => ({
      ...prev,
      currentMedications: prev.currentMedications.filter((_, i) => i !== index),
    }));
  };

  const updateMedication = (
    index: number,
    field: keyof Medication,
    value: string,
  ) => {
    setFormData((prev) => ({
      ...prev,
      currentMedications: prev.currentMedications.map((med, i) =>
        i === index ? { ...med, [field]: value } : med,
      ),
    }));
  };

  const addFamilyMember = () => {
    const newMember: FamilyHistory = {
      relation: "",
      conditions: [],
    };

    setFormData((prev) => ({
      ...prev,
      familyHistory: [...prev.familyHistory, newMember],
    }));
  };

  const removeFamilyMember = (index: number) => {
    setFormData((prev) => ({
      ...prev,
      familyHistory: prev.familyHistory.filter((_, i) => i !== index),
    }));
  };

  const handleNext = () => {
    setCurrentStep((prev) => Math.min(prev + 1, totalSteps));
  };

  const handleBack = () => {
    setCurrentStep((prev) => Math.max(prev - 1, 1));
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);

    try {
      const token = localStorage.getItem("healthcarePlus_token");

      const response = await fetch("/api/medical/history", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (response.ok) {
        updateOnboardingProgress("medicalHistory", true);
        setShowCompletion(true);
      } else {
        console.error("Failed to save medical history:", data);
        alert("Failed to save medical history. Please try again.");
      }
    } catch (error) {
      console.error("Error saving medical history:", error);
      alert("Network error. Please check your connection and try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCompletionClose = () => {
    setShowCompletion(false);
    navigate("/profile-setup");
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 1: // Current Health
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <Activity className="h-12 w-12 text-medical-green mx-auto mb-2" />
              <h3 className="text-lg font-semibold text-healthcare-text">
                Current Health Status
              </h3>
              <p className="text-muted-foreground">
                Tell us about your current medical conditions and medications
              </p>
            </div>

            {/* Current Conditions */}
            <div className="space-y-4">
              <Label className="text-healthcare-text font-medium text-base">
                Current Medical Conditions
              </Label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {commonConditions.map((condition) => (
                  <Button
                    key={condition}
                    variant="outline"
                    size="sm"
                    onClick={() => addCondition("current", condition)}
                    className="justify-start h-8 text-xs"
                  >
                    <Plus className="h-3 w-3 mr-1" />
                    {condition}
                  </Button>
                ))}
              </div>

              {formData.currentConditions.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium text-healthcare-text">
                    Selected conditions:
                  </p>
                  <div className="space-y-2">
                    {formData.currentConditions.map((condition, index) => (
                      <div
                        key={index}
                        className="flex items-center gap-2 p-2 bg-medical-light-green/20 rounded border"
                      >
                        <span className="flex-1 text-sm">
                          {condition.condition}
                        </span>
                        <Input
                          placeholder="Year diagnosed"
                          className="w-24 h-8 text-xs"
                          value={condition.diagnosedYear}
                          onChange={(e) => {
                            const newConditions = [
                              ...formData.currentConditions,
                            ];
                            newConditions[index].diagnosedYear = e.target.value;
                            setFormData((prev) => ({
                              ...prev,
                              currentConditions: newConditions,
                            }));
                          }}
                        />
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => removeCondition("current", index)}
                          className="h-6 w-6 p-0 text-red-600"
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Current Medications */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label className="text-healthcare-text font-medium text-base">
                  Current Medications
                </Label>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={addMedication}
                  className="text-medical-blue"
                >
                  <Plus className="h-4 w-4 mr-1" />
                  Add Medication
                </Button>
              </div>

              {formData.currentMedications.map((medication, index) => (
                <Card key={index} className="p-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <Label className="text-xs">Medication Name</Label>
                      <Input
                        placeholder="e.g., Lisinopril"
                        value={medication.name}
                        onChange={(e) =>
                          updateMedication(index, "name", e.target.value)
                        }
                        className="h-8 text-sm"
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Dosage</Label>
                      <Input
                        placeholder="e.g., 10mg"
                        value={medication.dosage}
                        onChange={(e) =>
                          updateMedication(index, "dosage", e.target.value)
                        }
                        className="h-8 text-sm"
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Frequency</Label>
                      <Select
                        value={medication.frequency}
                        onValueChange={(value) =>
                          updateMedication(index, "frequency", value)
                        }
                      >
                        <SelectTrigger className="h-8 text-sm">
                          <SelectValue placeholder="How often?" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="once-daily">Once daily</SelectItem>
                          <SelectItem value="twice-daily">
                            Twice daily
                          </SelectItem>
                          <SelectItem value="three-times-daily">
                            Three times daily
                          </SelectItem>
                          <SelectItem value="as-needed">As needed</SelectItem>
                          <SelectItem value="weekly">Weekly</SelectItem>
                          <SelectItem value="other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-end gap-2">
                      <div className="flex-1">
                        <Label className="text-xs">Prescribed By</Label>
                        <Input
                          placeholder="Doctor's name"
                          value={medication.prescribedBy}
                          onChange={(e) =>
                            updateMedication(
                              index,
                              "prescribedBy",
                              e.target.value,
                            )
                          }
                          className="h-8 text-sm"
                        />
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => removeMedication(index)}
                        className="h-8 w-8 p-0 text-red-600"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        );

      case 2: // Medical History
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <Stethoscope className="h-12 w-12 text-medical-blue mx-auto mb-2" />
              <h3 className="text-lg font-semibold text-healthcare-text">
                Past Medical History
              </h3>
              <p className="text-muted-foreground">
                Previous conditions, surgeries, and allergies
              </p>
            </div>

            {/* Past Conditions */}
            <div className="space-y-4">
              <Label className="text-healthcare-text font-medium text-base">
                Past Medical Conditions
              </Label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {commonConditions.map((condition) => (
                  <Button
                    key={condition}
                    variant="outline"
                    size="sm"
                    onClick={() => addCondition("past", condition)}
                    className="justify-start h-8 text-xs"
                  >
                    <Plus className="h-3 w-3 mr-1" />
                    {condition}
                  </Button>
                ))}
              </div>

              {formData.pastConditions.length > 0 && (
                <div className="space-y-2">
                  {formData.pastConditions.map((condition, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 p-2 bg-blue-50 rounded border"
                    >
                      <span className="flex-1 text-sm">
                        {condition.condition}
                      </span>
                      <Input
                        placeholder="Year diagnosed"
                        className="w-24 h-8 text-xs"
                        value={condition.diagnosedYear}
                        onChange={(e) => {
                          const newConditions = [...formData.pastConditions];
                          newConditions[index].diagnosedYear = e.target.value;
                          setFormData((prev) => ({
                            ...prev,
                            pastConditions: newConditions,
                          }));
                        }}
                      />
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => removeCondition("past", index)}
                        className="h-6 w-6 p-0 text-red-600"
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Allergies */}
            <div className="space-y-4">
              <Label className="text-healthcare-text font-medium text-base">
                Known Allergies
              </Label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label className="text-sm">Food Allergies</Label>
                  <Textarea
                    placeholder="List any food allergies..."
                    value={formData.allergies.foods.join(", ")}
                    onChange={(e) =>
                      setFormData((prev) => ({
                        ...prev,
                        allergies: {
                          ...prev.allergies,
                          foods: e.target.value
                            .split(",")
                            .map((item) => item.trim())
                            .filter(Boolean),
                        },
                      }))
                    }
                    className="min-h-[80px]"
                  />
                </div>
                <div>
                  <Label className="text-sm">Drug Allergies</Label>
                  <Textarea
                    placeholder="List any medication allergies..."
                    value={formData.allergies.medications.join(", ")}
                    onChange={(e) =>
                      setFormData((prev) => ({
                        ...prev,
                        allergies: {
                          ...prev.allergies,
                          medications: e.target.value
                            .split(",")
                            .map((item) => item.trim())
                            .filter(Boolean),
                        },
                      }))
                    }
                    className="min-h-[80px]"
                  />
                </div>
              </div>
            </div>
          </div>
        );

      case 3: // Family History
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <Users className="h-12 w-12 text-medical-green mx-auto mb-2" />
              <h3 className="text-lg font-semibold text-healthcare-text">
                Family Medical History
              </h3>
              <p className="text-muted-foreground">
                Health conditions in your immediate family
              </p>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label className="text-healthcare-text font-medium text-base">
                  Family Health History
                </Label>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={addFamilyMember}
                  className="text-medical-blue"
                >
                  <Plus className="h-4 w-4 mr-1" />
                  Add Family Member
                </Button>
              </div>

              {formData.familyHistory.map((member, index) => (
                <Card key={index} className="p-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Select
                        value={member.relation}
                        onValueChange={(value) => {
                          const newHistory = [...formData.familyHistory];
                          newHistory[index].relation = value;
                          setFormData((prev) => ({
                            ...prev,
                            familyHistory: newHistory,
                          }));
                        }}
                      >
                        <SelectTrigger className="w-48 h-8">
                          <SelectValue placeholder="Select relation" />
                        </SelectTrigger>
                        <SelectContent>
                          {familyRelations.map((relation) => (
                            <SelectItem key={relation} value={relation}>
                              {relation}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => removeFamilyMember(index)}
                        className="h-8 w-8 p-0 text-red-600"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>

                    <div>
                      <Label className="text-sm">Medical Conditions</Label>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-1 mt-1">
                        {commonConditions.map((condition) => (
                          <div
                            key={condition}
                            className="flex items-center space-x-1"
                          >
                            <Checkbox
                              id={`${index}-${condition}`}
                              checked={member.conditions.includes(condition)}
                              onCheckedChange={(checked) => {
                                const newHistory = [...formData.familyHistory];
                                if (checked) {
                                  newHistory[index].conditions = [
                                    ...newHistory[index].conditions,
                                    condition,
                                  ];
                                } else {
                                  newHistory[index].conditions = newHistory[
                                    index
                                  ].conditions.filter((c) => c !== condition);
                                }
                                setFormData((prev) => ({
                                  ...prev,
                                  familyHistory: newHistory,
                                }));
                              }}
                            />
                            <Label
                              htmlFor={`${index}-${condition}`}
                              className="text-xs cursor-pointer"
                            >
                              {condition}
                            </Label>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        );

      case 4: // Lifestyle
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <Activity className="h-12 w-12 text-medical-blue mx-auto mb-2" />
              <h3 className="text-lg font-semibold text-healthcare-text">
                Lifestyle Information
              </h3>
              <p className="text-muted-foreground">
                Your daily habits and lifestyle choices
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <Label className="text-healthcare-text font-medium">
                  Smoking Status
                </Label>
                <Select
                  value={formData.lifestyle.smoking}
                  onValueChange={(value) =>
                    setFormData((prev) => ({
                      ...prev,
                      lifestyle: { ...prev.lifestyle, smoking: value },
                    }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select smoking status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="never">Never smoked</SelectItem>
                    <SelectItem value="former">Former smoker</SelectItem>
                    <SelectItem value="current">Current smoker</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <Label className="text-healthcare-text font-medium">
                  Alcohol Consumption
                </Label>
                <Select
                  value={formData.lifestyle.alcohol}
                  onValueChange={(value) =>
                    setFormData((prev) => ({
                      ...prev,
                      lifestyle: { ...prev.lifestyle, alcohol: value },
                    }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select alcohol use" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="never">Never</SelectItem>
                    <SelectItem value="rarely">Rarely</SelectItem>
                    <SelectItem value="occasionally">Occasionally</SelectItem>
                    <SelectItem value="regularly">Regularly</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <Label className="text-healthcare-text font-medium">
                  Exercise Frequency
                </Label>
                <Select
                  value={formData.lifestyle.exercise}
                  onValueChange={(value) =>
                    setFormData((prev) => ({
                      ...prev,
                      lifestyle: { ...prev.lifestyle, exercise: value },
                    }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select exercise frequency" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">No regular exercise</SelectItem>
                    <SelectItem value="light">
                      Light (1-2 times/week)
                    </SelectItem>
                    <SelectItem value="moderate">
                      Moderate (3-4 times/week)
                    </SelectItem>
                    <SelectItem value="vigorous">
                      Vigorous (5+ times/week)
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <Label className="text-healthcare-text font-medium">
                  Diet Type
                </Label>
                <Select
                  value={formData.lifestyle.diet}
                  onValueChange={(value) =>
                    setFormData((prev) => ({
                      ...prev,
                      lifestyle: { ...prev.lifestyle, diet: value },
                    }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select diet type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="standard">Standard diet</SelectItem>
                    <SelectItem value="vegetarian">Vegetarian</SelectItem>
                    <SelectItem value="vegan">Vegan</SelectItem>
                    <SelectItem value="keto">Ketogenic</SelectItem>
                    <SelectItem value="low-carb">Low carb</SelectItem>
                    <SelectItem value="mediterranean">Mediterranean</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        );

      case 5: // Additional Information
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <AlertTriangle className="h-12 w-12 text-orange-500 mx-auto mb-2" />
              <h3 className="text-lg font-semibold text-healthcare-text">
                Additional Information
              </h3>
              <p className="text-muted-foreground">
                Any other important health information
              </p>
            </div>

            <div className="space-y-4">
              <div>
                <Label className="text-healthcare-text font-medium">
                  Recent Hospitalizations
                </Label>
                <Textarea
                  placeholder="Please describe any hospitalizations in the past 5 years..."
                  value={formData.additionalInfo.hospitalizations}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      additionalInfo: {
                        ...prev.additionalInfo,
                        hospitalizations: e.target.value,
                      },
                    }))
                  }
                  className="min-h-[100px]"
                />
              </div>

              <div>
                <Label className="text-healthcare-text font-medium">
                  Emergency Room Visits
                </Label>
                <Textarea
                  placeholder="Please describe any recent emergency room visits..."
                  value={formData.additionalInfo.emergencyRoomVisits}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      additionalInfo: {
                        ...prev.additionalInfo,
                        emergencyRoomVisits: e.target.value,
                      },
                    }))
                  }
                  className="min-h-[100px]"
                />
              </div>

              <div>
                <Label className="text-healthcare-text font-medium">
                  Significant Injuries
                </Label>
                <Textarea
                  placeholder="Please describe any significant injuries or accidents..."
                  value={formData.additionalInfo.significantInjuries}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      additionalInfo: {
                        ...prev.additionalInfo,
                        significantInjuries: e.target.value,
                      },
                    }))
                  }
                  className="min-h-[100px]"
                />
              </div>

              <div>
                <Label className="text-healthcare-text font-medium">
                  Other Health Concerns
                </Label>
                <Textarea
                  placeholder="Any other health concerns or information you'd like to share..."
                  value={formData.additionalInfo.otherConcerns}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      additionalInfo: {
                        ...prev.additionalInfo,
                        otherConcerns: e.target.value,
                      },
                    }))
                  }
                  className="min-h-[100px]"
                />
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light-blue via-background to-medical-light-green">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-medical-blue/20">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-medical-blue p-2 rounded-lg">
              <Heart className="h-6 w-6 text-white" />
            </div>
            <span className="text-xl font-bold text-healthcare-text">
              HealthCare Plus
            </span>
          </div>
          <Button
            variant="ghost"
            onClick={() => navigate("/profile-setup")}
            className="text-healthcare-text hover:text-medical-blue"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Onboarding
          </Button>
        </div>
      </header>

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Page Header */}
          <div className="text-center mb-8">
            <div className="bg-medical-green/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Stethoscope className="h-8 w-8 text-medical-green" />
            </div>
            <h1 className="text-3xl font-bold text-healthcare-text mb-2">
              Medical History
            </h1>
            <p className="text-lg text-muted-foreground">
              Step {currentStep} of {totalSteps}: {stepTitles[currentStep - 1]}
            </p>
          </div>

          {/* Progress Bar */}
          <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm mb-8">
            <CardContent className="p-6">
              <div className="space-y-2">
                <Progress value={progressPercentage} className="w-full h-3" />
                <div className="flex justify-between text-xs text-muted-foreground">
                  {stepTitles.map((title, index) => (
                    <span
                      key={index}
                      className={`${index + 1 <= currentStep ? "text-medical-blue font-medium" : ""}`}
                    >
                      {index + 1}. {title}
                    </span>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Form Content */}
          <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
            <CardContent className="p-8">{renderStepContent()}</CardContent>
          </Card>

          {/* Navigation Buttons */}
          <div className="flex justify-between mt-8">
            <Button
              variant="outline"
              onClick={handleBack}
              disabled={currentStep === 1}
              className="border-medical-blue text-medical-blue hover:bg-medical-blue hover:text-white"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>

            {currentStep < totalSteps ? (
              <Button
                onClick={handleNext}
                className="bg-medical-blue hover:bg-medical-blue/90 text-white"
              >
                Continue
                <ArrowLeft className="ml-2 h-4 w-4 rotate-180" />
              </Button>
            ) : (
              <Button
                onClick={handleSubmit}
                disabled={isSubmitting}
                className="bg-medical-green hover:bg-medical-green/90 text-white"
              >
                {isSubmitting ? (
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Saving...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4" />
                    Complete Medical History
                  </div>
                )}
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Completion Modal */}
      <Dialog open={showCompletion} onOpenChange={setShowCompletion}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader className="text-center">
            <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle2 className="h-8 w-8 text-green-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              Medical History Complete!
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Your medical history has been successfully saved. This information
              will help us provide you with the best possible care.
            </DialogDescription>
          </DialogHeader>

          <div className="flex justify-center gap-3 mt-6">
            <Button
              onClick={handleCompletionClose}
              className="bg-medical-blue hover:bg-medical-blue/90"
            >
              Continue Onboarding
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
