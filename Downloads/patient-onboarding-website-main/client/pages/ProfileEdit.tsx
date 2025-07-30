import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useUser, UserData } from "@/lib/UserContext";
import {
  Heart,
  ArrowLeft,
  User,
  Mail,
  Phone,
  Calendar,
  CheckCircle2,
  Save,
} from "lucide-react";

export default function ProfileEdit() {
  const navigate = useNavigate();
  const { userData, setUserData } = useUser();
  const [showSuccess, setShowSuccess] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [formErrors, setFormErrors] = useState<{ [key: string]: string }>({});

  const [formData, setFormData] = useState<UserData>({
    firstName: "",
    lastName: "",
    email: "",
    phone: "",
    dateOfBirth: "",
    gender: "",
  });

  // Load user data when component mounts
  useEffect(() => {
    if (userData) {
      setFormData(userData);
    }
  }, [userData]);

  const handleInputChange = (field: keyof UserData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (formErrors[field]) {
      setFormErrors((prev) => ({ ...prev, [field]: "" }));
    }
  };

  const validateForm = (): boolean => {
    const errors: { [key: string]: string } = {};

    if (!formData.firstName.trim()) {
      errors.firstName = "First name is required";
    }

    if (!formData.lastName.trim()) {
      errors.lastName = "Last name is required";
    }

    if (!formData.email.trim()) {
      errors.email = "Email is required";
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      errors.email = "Email is invalid";
    }

    if (!formData.phone.trim()) {
      errors.phone = "Phone number is required";
    }

    if (!formData.dateOfBirth) {
      errors.dateOfBirth = "Date of birth is required";
    }

    if (!formData.gender) {
      errors.gender = "Gender is required";
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateForm()) return;

    setIsLoading(true);

    // Simulate save process
    setTimeout(() => {
      setIsLoading(false);
      setUserData(formData);
      setShowSuccess(true);
    }, 1500);
  };

  const handleSuccessClose = () => {
    setShowSuccess(false);
    navigate("/dashboard");
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
            onClick={() => navigate("/dashboard")}
            className="text-healthcare-text hover:text-medical-blue"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Button>
        </div>
      </header>

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-2xl mx-auto">
          {/* Page Header */}
          <div className="text-center mb-8">
            <div className="bg-medical-blue/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <User className="h-8 w-8 text-medical-blue" />
            </div>
            <h1 className="text-3xl font-bold text-healthcare-text mb-2">
              Edit Profile
            </h1>
            <p className="text-lg text-muted-foreground">
              Update your personal information
            </p>
          </div>

          {/* Profile Form */}
          <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-xl text-healthcare-text">
                Personal Information
              </CardTitle>
              <CardDescription>
                Keep your information up to date for the best healthcare
                experience
              </CardDescription>
            </CardHeader>

            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Name Fields */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label
                      htmlFor="firstName"
                      className="text-healthcare-text font-medium"
                    >
                      First Name *
                    </Label>
                    <div className="relative">
                      <User className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                      <Input
                        id="firstName"
                        type="text"
                        placeholder="First name"
                        value={formData.firstName}
                        onChange={(e) =>
                          handleInputChange("firstName", e.target.value)
                        }
                        className={`pl-10 h-12 ${formErrors.firstName ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                        required
                      />
                    </div>
                    {formErrors.firstName && (
                      <p className="text-red-500 text-sm">
                        {formErrors.firstName}
                      </p>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label
                      htmlFor="lastName"
                      className="text-healthcare-text font-medium"
                    >
                      Last Name *
                    </Label>
                    <div className="relative">
                      <User className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                      <Input
                        id="lastName"
                        type="text"
                        placeholder="Last name"
                        value={formData.lastName}
                        onChange={(e) =>
                          handleInputChange("lastName", e.target.value)
                        }
                        className={`pl-10 h-12 ${formErrors.lastName ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                        required
                      />
                    </div>
                    {formErrors.lastName && (
                      <p className="text-red-500 text-sm">
                        {formErrors.lastName}
                      </p>
                    )}
                  </div>
                </div>

                {/* Contact Information */}
                <div className="space-y-2">
                  <Label
                    htmlFor="email"
                    className="text-healthcare-text font-medium"
                  >
                    Email Address *
                  </Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="your.email@example.com"
                      value={formData.email}
                      onChange={(e) =>
                        handleInputChange("email", e.target.value)
                      }
                      className={`pl-10 h-12 ${formErrors.email ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                      required
                    />
                  </div>
                  {formErrors.email && (
                    <p className="text-red-500 text-sm">{formErrors.email}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label
                    htmlFor="phone"
                    className="text-healthcare-text font-medium"
                  >
                    Phone Number *
                  </Label>
                  <div className="relative">
                    <Phone className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="phone"
                      type="tel"
                      placeholder="(555) 123-4567"
                      value={formData.phone}
                      onChange={(e) =>
                        handleInputChange("phone", e.target.value)
                      }
                      className={`pl-10 h-12 ${formErrors.phone ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-medical-blue/30 focus:border-medical-blue focus:ring-medical-blue"}`}
                      required
                    />
                  </div>
                  {formErrors.phone && (
                    <p className="text-red-500 text-sm">{formErrors.phone}</p>
                  )}
                </div>

                {/* Additional Information */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label
                      htmlFor="dateOfBirth"
                      className="text-healthcare-text font-medium"
                    >
                      Date of Birth *
                    </Label>
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

                  <div className="space-y-2">
                    <Label
                      htmlFor="gender"
                      className="text-healthcare-text font-medium"
                    >
                      Gender *
                    </Label>
                    <Select
                      value={formData.gender}
                      onValueChange={(value) =>
                        handleInputChange("gender", value)
                      }
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
                      <p className="text-red-500 text-sm">
                        {formErrors.gender}
                      </p>
                    )}
                  </div>
                </div>

                {/* Submit Button */}
                <div className="flex gap-4 pt-6">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => navigate("/dashboard")}
                    className="flex-1 border-gray-300 text-gray-600 hover:bg-gray-50"
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    disabled={isLoading}
                    className="flex-1 bg-medical-blue hover:bg-medical-blue/90 text-white"
                  >
                    {isLoading ? (
                      <div className="flex items-center gap-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        Saving...
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <Save className="h-4 w-4" />
                        Save Changes
                      </div>
                    )}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Success Modal */}
      <Dialog open={showSuccess} onOpenChange={setShowSuccess}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader className="text-center">
            <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle2 className="h-8 w-8 text-green-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text">
              Profile Updated!
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Your profile information has been successfully updated. The
              changes are now reflected in your account.
            </DialogDescription>
          </DialogHeader>

          <div className="flex justify-center gap-3 mt-6">
            <Button
              onClick={handleSuccessClose}
              className="bg-medical-blue hover:bg-medical-blue/90"
            >
              Return to Dashboard
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
