import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
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
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import {
  Heart,
  ArrowLeft,
  Mail,
  Lock,
  Eye,
  EyeOff,
  User,
  Phone,
  Calendar,
  CheckCircle2,
  ArrowRight,
  Info,
  Shield,
  Activity,
  FileText,
  X,
} from "lucide-react";

interface FormData {
  // Personal Details
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  dateOfBirth: string;
  gender: string;

  // Account Security
  password: string;
  confirmPassword: string;

  // Health Basics
  bloodType: string;
  allergies: string;
  currentMedications: string[];

  // Insurance Info
  insuranceProvider: string;
  policyNumber: string;
  groupNumber: string;
}

interface FormErrors {
  [key: string]: string;
}

export default function Register() {
  const navigate = useNavigate();
  const { setUserData } = useUser();
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState<FormData>({
    firstName: "",
    lastName: "",
    email: "",
    phone: "",
    dateOfBirth: "",
    gender: "",
    password: "",
    confirmPassword: "",
    bloodType: "",
    allergies: "",
    currentMedications: [],
    insuranceProvider: "",
    policyNumber: "",
    groupNumber: "",
  });
  const [formErrors, setFormErrors] = useState<FormErrors>({});
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [agreedToTerms, setAgreedToTerms] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [newMedication, setNewMedication] = useState("");

  const totalSteps = 4;
  const progressPercentage = (currentStep / totalSteps) * 100;

  const stepTitles = [
    "Personal Details",
    "Account Security",
    "Health Information",
    "Insurance & Finish",
  ];

  const bloodTypes = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"];

  const commonMedications = [
    "Aspirin",
    "Ibuprofen",
    "Acetaminophen",
    "Lisinopril",
    "Metformin",
    "Amlodipine",
    "Metoprolol",
    "Omeprazole",
    "Simvastatin",
    "Losartan",
  ];

  const handleInputChange = (field: keyof FormData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (formErrors[field]) {
      setFormErrors((prev) => ({ ...prev, [field]: "" }));
    }
  };

  const addMedication = (medication: string) => {
    if (medication && !formData.currentMedications.includes(medication)) {
      setFormData((prev) => ({
        ...prev,
        currentMedications: [...prev.currentMedications, medication],
      }));
      setNewMedication("");
    }
  };

  const removeMedication = (medication: string) => {
    setFormData((prev) => ({
      ...prev,
      currentMedications: prev.currentMedications.filter(
        (med) => med !== medication,
      ),
    }));
  };

  const getPasswordStrength = (password: string) => {
    let score = 0;
    if (password.length >= 8) score++;
    if (/[a-z]/.test(password)) score++;
    if (/[A-Z]/.test(password)) score++;
    if (/\d/.test(password)) score++;
    if (/[^a-zA-Z\d]/.test(password)) score++;

    const strengths = ["Very Weak", "Weak", "Fair", "Good", "Strong"];
    const colors = [
      "bg-red-500",
      "bg-orange-500",
      "bg-yellow-500",
      "bg-blue-500",
      "bg-green-500",
    ];

    return {
      score,
      text: strengths[score] || "Very Weak",
      color: colors[score] || "bg-red-500",
      percentage: (score / 5) * 100,
    };
  };

  const validateStep = (step: number): boolean => {
    const errors: FormErrors = {};

    switch (step) {
      case 1:
        if (!formData.firstName.trim())
          errors.firstName = "First name is required";
        if (!formData.lastName.trim())
          errors.lastName = "Last name is required";
        if (!formData.email.trim()) errors.email = "Email is required";
        else if (!/\S+@\S+\.\S+/.test(formData.email))
          errors.email = "Email is invalid";
        if (!formData.phone.trim()) errors.phone = "Phone number is required";
        if (!formData.dateOfBirth)
          errors.dateOfBirth = "Date of birth is required";
        if (!formData.gender) errors.gender = "Gender is required";
        break;

      case 2:
        if (!formData.password) errors.password = "Password is required";
        else if (formData.password.length < 8)
          errors.password = "Password must be at least 8 characters";
        if (!formData.confirmPassword)
          errors.confirmPassword = "Please confirm your password";
        else if (formData.password !== formData.confirmPassword)
          errors.confirmPassword = "Passwords do not match";
        break;

      case 3:
        if (!formData.bloodType) errors.bloodType = "Blood type is required";
        break;

      case 4:
        if (!agreedToTerms)
          errors.terms = "You must agree to the terms and conditions";
        break;
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep((prev) => Math.min(prev + 1, totalSteps));
    }
  };

  const handleBack = () => {
    setCurrentStep((prev) => Math.max(prev - 1, 1));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateStep(4)) return;

    setIsLoading(true);

    try {
      const response = await fetch("/api/auth/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          firstName: formData.firstName,
          lastName: formData.lastName,
          email: formData.email,
          phone: formData.phone,
          dateOfBirth: formData.dateOfBirth,
          gender: formData.gender,
          password: formData.password,
          bloodType: formData.bloodType,
          allergies: formData.allergies,
          currentMedications: formData.currentMedications,
          insuranceProvider: formData.insuranceProvider,
          policyNumber: formData.policyNumber,
          groupNumber: formData.groupNumber,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Store JWT token
        localStorage.setItem("healthcarePlus_token", data.token);

        // Save user data to context
        setUserData({
          firstName: data.user.firstName,
          lastName: data.user.lastName,
          email: data.user.email,
          phone: data.user.phone,
          dateOfBirth: data.user.dateOfBirth,
          gender: data.user.gender,
          role: data.user.role,
        });

        navigate("/profile-setup");
      } else {
        // Handle registration errors
        if (response.status === 409) {
          setFormErrors({ email: data.message || "Email already exists" });
          setCurrentStep(1); // Go back to first step to show email error
        } else {
          setFormErrors({ general: data.message || "Registration failed" });
        }
      }
    } catch (error) {
      console.error("Registration error:", error);
      setFormErrors({ general: "email exists" });
    } finally {
      setIsLoading(false);
    }
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label
                    htmlFor="firstName"
                    className="text-healthcare-text font-medium"
                  >
                    First Name
                  </Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Your legal first name as it appears on your ID</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <div className="relative">
                  <User className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="firstName"
                    type="text"
                    placeholder="John"
                    value={formData.firstName}
                    onChange={(e) =>
                      handleInputChange("firstName", e.target.value)
                    }
                    className={`pl-10 h-12 ${formErrors.firstName ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                    required
                  />
                </div>
                {formErrors.firstName && (
                  <p className="text-red-500 text-sm">{formErrors.firstName}</p>
                )}
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label
                    htmlFor="lastName"
                    className="text-healthcare-text font-medium"
                  >
                    Last Name
                  </Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Your legal last name as it appears on your ID</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <div className="relative">
                  <User className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="lastName"
                    type="text"
                    placeholder="Doe"
                    value={formData.lastName}
                    onChange={(e) =>
                      handleInputChange("lastName", e.target.value)
                    }
                    className={`pl-10 h-12 ${formErrors.lastName ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                    required
                  />
                </div>
                {formErrors.lastName && (
                  <p className="text-red-500 text-sm">{formErrors.lastName}</p>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label
                  htmlFor="email"
                  className="text-healthcare-text font-medium"
                >
                  Email Address
                </Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>
                        We'll use this to send appointment reminders and health
                        updates
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <div className="relative">
                <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="email"
                  type="email"
                  placeholder="john.doe@example.com"
                  value={formData.email}
                  onChange={(e) => handleInputChange("email", e.target.value)}
                  className={`pl-10 h-12 ${formErrors.email ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                  required
                />
              </div>
              {formErrors.email && (
                <p className="text-red-500 text-sm">{formErrors.email}</p>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label
                    htmlFor="phone"
                    className="text-healthcare-text font-medium"
                  >
                    Phone Number
                  </Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>
                          For urgent communications and appointment
                          confirmations
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <div className="relative">
                  <Phone className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="phone"
                    type="tel"
                    placeholder="(555) 123-4567"
                    value={formData.phone}
                    onChange={(e) => handleInputChange("phone", e.target.value)}
                    className={`pl-10 h-12 ${formErrors.phone ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                    required
                  />
                </div>
                {formErrors.phone && (
                  <p className="text-red-500 text-sm">{formErrors.phone}</p>
                )}
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label
                    htmlFor="dateOfBirth"
                    className="text-healthcare-text font-medium"
                  >
                    Date of Birth
                  </Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>
                          Required for age-appropriate care and medication
                          dosing
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <div className="relative">
                  <Calendar className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="dateOfBirth"
                    type="date"
                    value={formData.dateOfBirth}
                    onChange={(e) =>
                      handleInputChange("dateOfBirth", e.target.value)
                    }
                    className={`pl-10 h-12 ${formErrors.dateOfBirth ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                    required
                  />
                </div>
                {formErrors.dateOfBirth && (
                  <p className="text-red-500 text-sm">
                    {formErrors.dateOfBirth}
                  </p>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label
                  htmlFor="gender"
                  className="text-healthcare-text font-medium"
                >
                  Gender
                </Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>
                        Helps determine appropriate health screenings and risk
                        factors
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Select
                value={formData.gender}
                onValueChange={(value) => handleInputChange("gender", value)}
              >
                <SelectTrigger
                  className={`h-12 ${formErrors.gender ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                >
                  <SelectValue placeholder="Select your gender" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="male">Male</SelectItem>
                  <SelectItem value="female">Female</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                  <SelectItem value="prefer-not-to-say">
                    Prefer not to say
                  </SelectItem>
                </SelectContent>
              </Select>
              {formErrors.gender && (
                <p className="text-red-500 text-sm">{formErrors.gender}</p>
              )}
            </div>
          </div>
        );

      case 2:
        const passwordStrength = getPasswordStrength(formData.password);
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <Shield className="h-12 w-12 text-medical-blue mx-auto mb-2" />
              <h3 className="text-lg font-semibold text-healthcare-text">
                Secure Your Account
              </h3>
              <p className="text-muted-foreground">
                Create a strong password to protect your health information
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label
                  htmlFor="password"
                  className="text-healthcare-text font-medium"
                >
                  Password
                </Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>
                        Use at least 8 characters with uppercase, lowercase,
                        numbers, and symbols
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Create a strong password"
                  value={formData.password}
                  onChange={(e) =>
                    handleInputChange("password", e.target.value)
                  }
                  className={`pl-10 pr-10 h-12 ${formErrors.password ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-3 text-muted-foreground hover:text-healthcare-text"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
              {formErrors.password && (
                <p className="text-red-500 text-sm">{formErrors.password}</p>
              )}

              {/* Password Strength Meter */}
              {formData.password && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">
                      Password Strength:
                    </span>
                    <span
                      className={`text-sm font-medium ${passwordStrength.score >= 3 ? "text-green-600" : passwordStrength.score >= 2 ? "text-yellow-600" : "text-red-600"}`}
                    >
                      {passwordStrength.text}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-300 ${passwordStrength.color}`}
                      style={{ width: `${passwordStrength.percentage}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-2">
              <Label
                htmlFor="confirmPassword"
                className="text-healthcare-text font-medium"
              >
                Confirm Password
              </Label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="confirmPassword"
                  type={showConfirmPassword ? "text" : "password"}
                  placeholder="Confirm your password"
                  value={formData.confirmPassword}
                  onChange={(e) =>
                    handleInputChange("confirmPassword", e.target.value)
                  }
                  className={`pl-10 pr-10 h-12 ${formErrors.confirmPassword ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-3 text-muted-foreground hover:text-healthcare-text"
                >
                  {showConfirmPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
              {formErrors.confirmPassword && (
                <p className="text-red-500 text-sm">
                  {formErrors.confirmPassword}
                </p>
              )}
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <Activity className="h-12 w-12 text-medical-green mx-auto mb-2" />
              <h3 className="text-lg font-semibold text-healthcare-text">
                Health Information
              </h3>
              <p className="text-muted-foreground">
                This information helps us provide better care
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label
                  htmlFor="bloodType"
                  className="text-healthcare-text font-medium"
                >
                  Blood Type
                </Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>
                        Critical information for emergency situations and blood
                        transfusions
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Select
                value={formData.bloodType}
                onValueChange={(value) => handleInputChange("bloodType", value)}
              >
                <SelectTrigger
                  className={`h-12 ${formErrors.bloodType ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                >
                  <SelectValue placeholder="Select your blood type" />
                </SelectTrigger>
                <SelectContent>
                  {bloodTypes.map((type) => (
                    <SelectItem key={type} value={type}>
                      {type}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {formErrors.bloodType && (
                <p className="text-red-500 text-sm">{formErrors.bloodType}</p>
              )}
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label
                  htmlFor="allergies"
                  className="text-healthcare-text font-medium"
                >
                  Known Allergies
                </Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>
                        Include food, drug, and environmental allergies to
                        prevent adverse reactions
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Textarea
                id="allergies"
                placeholder="List any known allergies (e.g., penicillin, peanuts, latex)..."
                value={formData.allergies}
                onChange={(e) => handleInputChange("allergies", e.target.value)}
                className="min-h-[100px] border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label className="text-healthcare-text font-medium">
                  Current Medications
                </Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>
                        Include all prescription and over-the-counter
                        medications to avoid interactions
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>

              <div className="space-y-3">
                <div className="flex gap-2">
                  <Input
                    placeholder="Add a medication..."
                    value={newMedication}
                    onChange={(e) => setNewMedication(e.target.value)}
                    className="flex-1 border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"
                    onKeyPress={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        addMedication(newMedication);
                      }
                    }}
                  />
                  <Button
                    type="button"
                    onClick={() => addMedication(newMedication)}
                    className="bg-medical-green hover:bg-medical-green/90"
                  >
                    Add
                  </Button>
                </div>

                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Common medications:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {commonMedications.map((med) => (
                      <Badge
                        key={med}
                        variant="outline"
                        className="cursor-pointer hover:bg-medical-light-blue border-medical-blue text-medical-blue"
                        onClick={() => addMedication(med)}
                      >
                        {med}
                      </Badge>
                    ))}
                  </div>
                </div>

                {formData.currentMedications.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-healthcare-text">
                      Selected medications:
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {formData.currentMedications.map((med) => (
                        <Badge
                          key={med}
                          className="bg-medical-blue text-white flex items-center gap-1"
                        >
                          {med}
                          <X
                            className="h-3 w-3 cursor-pointer hover:text-red-200"
                            onClick={() => removeMedication(med)}
                          />
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <FileText className="h-12 w-12 text-medical-blue mx-auto mb-2" />
              <h3 className="text-lg font-semibold text-healthcare-text">
                Insurance Information
              </h3>
              <p className="text-muted-foreground">
                Optional - helps us verify your coverage
              </p>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label
                    htmlFor="insuranceProvider"
                    className="text-healthcare-text font-medium"
                  >
                    Insurance Provider (Optional)
                  </Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>
                          The name of your insurance company (e.g., Blue Cross,
                          Aetna)
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Input
                  id="insuranceProvider"
                  type="text"
                  placeholder="e.g., Blue Cross Blue Shield"
                  value={formData.insuranceProvider}
                  onChange={(e) =>
                    handleInputChange("insuranceProvider", e.target.value)
                  }
                  className="h-12 border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label
                    htmlFor="policyNumber"
                    className="text-healthcare-text font-medium"
                  >
                    Policy Number (Optional)
                  </Label>
                  <Input
                    id="policyNumber"
                    type="text"
                    placeholder="Policy number"
                    value={formData.policyNumber}
                    onChange={(e) =>
                      handleInputChange("policyNumber", e.target.value)
                    }
                    className="h-12 border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"
                  />
                </div>

                <div className="space-y-2">
                  <Label
                    htmlFor="groupNumber"
                    className="text-healthcare-text font-medium"
                  >
                    Group Number (Optional)
                  </Label>
                  <Input
                    id="groupNumber"
                    type="text"
                    placeholder="Group number"
                    value={formData.groupNumber}
                    onChange={(e) =>
                      handleInputChange("groupNumber", e.target.value)
                    }
                    className="h-12 border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"
                  />
                </div>
              </div>
            </div>

            <div className="bg-medical-light-blue/20 p-4 rounded-lg space-y-4">
              <div className="flex items-start space-x-2">
                <Checkbox
                  id="terms"
                  checked={agreedToTerms}
                  onCheckedChange={(checked) =>
                    setAgreedToTerms(checked as boolean)
                  }
                  className="mt-1"
                />
                <Label
                  htmlFor="terms"
                  className="text-sm text-healthcare-text cursor-pointer leading-relaxed"
                >
                  I agree to the{" "}
                  <Link
                    to="/terms"
                    className="text-medical-blue hover:underline font-medium"
                  >
                    Terms of Service
                  </Link>{" "}
                  and{" "}
                  <Link
                    to="/privacy"
                    className="text-medical-blue hover:underline font-medium"
                  >
                    Privacy Policy
                  </Link>
                  . I understand that my information will be used in accordance
                  with HIPAA regulations and consent to electronic
                  communications regarding my healthcare.
                </Label>
              </div>
              {formErrors.terms && (
                <p className="text-red-500 text-sm">{formErrors.terms}</p>
              )}
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
            onClick={() => navigate("/")}
            className="text-healthcare-text hover:text-medical-blue"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Button>
        </div>
      </header>

      {/* Registration Form */}
      <div className="container mx-auto px-4 py-8 flex items-center justify-center">
        <Card className="w-full max-w-3xl shadow-2xl border-0 bg-white/90 backdrop-blur-sm">
          <CardHeader className="text-center space-y-4">
            <div className="bg-medical-green/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto">
              <User className="h-8 w-8 text-medical-green" />
            </div>
            <CardTitle className="text-2xl font-bold text-healthcare-text">
              Create Your Account
            </CardTitle>
            <CardDescription className="text-base">
              Step {currentStep} of {totalSteps}: {stepTitles[currentStep - 1]}
            </CardDescription>

            {/* Progress Bar */}
            <div className="space-y-2">
              <Progress value={progressPercentage} className="w-full h-2" />
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
          </CardHeader>

          <CardContent className="space-y-6">
            <form
              onSubmit={
                currentStep === totalSteps
                  ? handleSubmit
                  : (e) => e.preventDefault()
              }
            >
              {renderStepContent()}

              {/* General Error Display */}
              {formErrors.general && (
                <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-800 text-sm font-medium">
                    {formErrors.general}
                  </p>
                </div>
              )}

              {/* Navigation Buttons */}
              <div className="flex justify-between pt-6">
                <Button
                  type="button"
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
                    type="button"
                    onClick={handleNext}
                    className="bg-medical-blue hover:bg-medical-blue/90 text-white"
                  >
                    Save and Continue
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                ) : (
                  <Button
                    type="submit"
                    disabled={isLoading || !agreedToTerms}
                    className="bg-medical-green hover:bg-medical-green/90 text-white"
                  >
                    {isLoading ? (
                      <div className="flex items-center gap-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        Creating Account...
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4" />
                        Complete Registration
                      </div>
                    )}
                  </Button>
                )}
              </div>
            </form>

            {/* Already have account link */}
            <div className="text-center pt-4 border-t border-medical-blue/20">
              <p className="text-sm text-muted-foreground">
                Already have an account?{" "}
                <Link
                  to="/login"
                  className="text-medical-blue hover:underline font-medium"
                >
                  Sign in here
                </Link>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
