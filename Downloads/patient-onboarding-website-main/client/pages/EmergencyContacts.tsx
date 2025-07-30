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
import { Badge } from "@/components/ui/badge";
import {
  Heart,
  ArrowLeft,
  User,
  Phone,
  Mail,
  MapPin,
  Plus,
  X,
  CheckCircle2,
  Users,
  AlertTriangle,
} from "lucide-react";

interface EmergencyContact {
  id: string;
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

export default function EmergencyContacts() {
  const navigate = useNavigate();
  const { updateOnboardingProgress } = useUser();
  const [contacts, setContacts] = useState<EmergencyContact[]>([]);
  const [showCompletion, setShowCompletion] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formErrors, setFormErrors] = useState<{
    [contactId: string]: { [field: string]: string };
  }>({});

  const relationshipOptions = [
    "Spouse",
    "Partner",
    "Parent",
    "Child",
    "Sibling",
    "Grandparent",
    "Grandchild",
    "Aunt",
    "Uncle",
    "Cousin",
    "Friend",
    "Neighbor",
    "Other Family Member",
    "Other",
  ];

  const stateOptions = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
  ];

  const addContact = () => {
    const newContact: EmergencyContact = {
      id: `contact-${Date.now()}`,
      firstName: "",
      lastName: "",
      relationship: "",
      primaryPhone: "",
      email: "",
      address: {
        street: "",
        city: "",
        state: "",
        zipCode: "",
      },
      isAuthorizedToReceiveInfo: false,
      canMakeHealthcareDecisions: false,
    };

    setContacts((prev) => [...prev, newContact]);
  };

  const removeContact = (contactId: string) => {
    setContacts((prev) => prev.filter((contact) => contact.id !== contactId));
    setFormErrors((prev) => {
      const newErrors = { ...prev };
      delete newErrors[contactId];
      return newErrors;
    });
  };

  const updateContact = (
    contactId: string,
    field: string,
    value: string | boolean,
  ) => {
    setContacts((prev) =>
      prev.map((contact) => {
        if (contact.id === contactId) {
          if (field.includes(".")) {
            const [parentField, childField] = field.split(".");
            return {
              ...contact,
              [parentField]: {
                ...(contact[parentField as keyof EmergencyContact] as any),
                [childField]: value,
              },
            };
          }
          return { ...contact, [field]: value };
        }
        return contact;
      }),
    );

    // Clear error when user starts typing
    if (formErrors[contactId]?.[field]) {
      setFormErrors((prev) => ({
        ...prev,
        [contactId]: {
          ...prev[contactId],
          [field]: "",
        },
      }));
    }
  };

  const validateContacts = (): boolean => {
    const errors: { [contactId: string]: { [field: string]: string } } = {};
    let hasErrors = false;

    contacts.forEach((contact) => {
      const contactErrors: { [field: string]: string } = {};

      if (!contact.firstName.trim()) {
        contactErrors.firstName = "First name is required";
        hasErrors = true;
      }

      if (!contact.lastName.trim()) {
        contactErrors.lastName = "Last name is required";
        hasErrors = true;
      }

      if (!contact.relationship) {
        contactErrors.relationship = "Relationship is required";
        hasErrors = true;
      }

      if (!contact.primaryPhone.trim()) {
        contactErrors.primaryPhone = "Primary phone number is required";
        hasErrors = true;
      } else {
        // 10-digit phone number validation
        const digits = contact.primaryPhone.replace(/\D/g, "");
        if (digits.length !== 10) {
          contactErrors.primaryPhone =
            "Please enter a valid 10-digit phone number";
          hasErrors = true;
        }
      }

      if (contact.email && !/\S+@\S+\.\S+/.test(contact.email)) {
        contactErrors.email = "Please enter a valid email address";
        hasErrors = true;
      }

      if (Object.keys(contactErrors).length > 0) {
        errors[contact.id] = contactErrors;
      }
    });

    setFormErrors(errors);
    return !hasErrors;
  };

  const handlePhoneInput = (
    contactId: string,
    field: string,
    value: string,
  ) => {
    // Allow only numbers, limit to 10 digits
    const phoneNumber = value.replace(/\D/g, "").slice(0, 10);
    updateContact(contactId, field, phoneNumber);
  };

  const canSubmit = () => {
    return contacts.length >= 1 && validateContacts();
  };

  const handleSubmit = async () => {
    if (!validateContacts()) return;

    setIsSubmitting(true);

    try {
      const token = localStorage.getItem("healthcarePlus_token");

      const response = await fetch("/api/medical/emergency-contacts", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify({ contacts }),
      });

      const data = await response.json();

      if (response.ok) {
        updateOnboardingProgress("emergencyContacts", true);
        setShowCompletion(true);
      } else {
        console.error("Failed to save emergency contacts:", data);
        alert("Failed to save emergency contacts. Please try again.");
      }
    } catch (error) {
      console.error("Error saving emergency contacts:", error);
      alert("Network error. Please check your connection and try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCompletionClose = () => {
    setShowCompletion(false);
    navigate("/profile-setup");
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
            <div className="bg-medical-blue/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Users className="h-8 w-8 text-medical-blue" />
            </div>
            <h1 className="text-3xl font-bold text-healthcare-text mb-2">
              Emergency Contacts
            </h1>
            <p className="text-lg text-muted-foreground">
              Add contact information for emergencies and healthcare decisions
            </p>
          </div>

          {/* Instructions */}
          <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm mb-8">
            <CardHeader>
              <CardTitle className="text-lg text-healthcare-text flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-orange-500" />
                Important Information
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  • Please add at least one emergency contact who can be reached
                  24/7
                </p>
                <p>
                  • Emergency contacts may be contacted in case of medical
                  emergencies
                </p>
                <p>
                  • You can authorize specific contacts to receive health
                  information
                </p>
                <p>
                  • Healthcare decision-making authority requires legal
                  documentation
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Add Contact Button */}
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-healthcare-text">
              Emergency Contacts ({contacts.length})
            </h2>
            <Button
              onClick={addContact}
              className="bg-medical-blue hover:bg-medical-blue/90 text-white"
            >
              <Plus className="mr-2 h-4 w-4" />
              Add Emergency Contact
            </Button>
          </div>

          {/* Contacts List */}
          {contacts.length === 0 ? (
            <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
              <CardContent className="p-8 text-center">
                <Users className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-healthcare-text mb-2">
                  No Emergency Contacts Added
                </h3>
                <p className="text-muted-foreground mb-4">
                  Add at least one emergency contact to continue with your
                  onboarding.
                </p>
                <Button
                  onClick={addContact}
                  className="bg-medical-blue hover:bg-medical-blue/90 text-white"
                >
                  <Plus className="mr-2 h-4 w-4" />
                  Add Your First Contact
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-6">
              {contacts.map((contact, index) => (
                <Card
                  key={contact.id}
                  className="border-0 shadow-lg bg-white/90 backdrop-blur-sm"
                >
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg text-healthcare-text">
                        Emergency Contact {index + 1}
                      </CardTitle>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeContact(contact.id)}
                        className="text-red-600 hover:text-red-700 hover:bg-red-50"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-6">
                    {/* Basic Information */}
                    <div>
                      <h4 className="font-medium text-healthcare-text mb-3">
                        Basic Information
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor={`${contact.id}-firstName`}>
                            First Name *
                          </Label>
                          <Input
                            id={`${contact.id}-firstName`}
                            value={contact.firstName}
                            onChange={(e) =>
                              updateContact(
                                contact.id,
                                "firstName",
                                e.target.value,
                              )
                            }
                            className={
                              formErrors[contact.id]?.firstName
                                ? "border-red-500"
                                : ""
                            }
                            placeholder="First name"
                          />
                          {formErrors[contact.id]?.firstName && (
                            <p className="text-red-500 text-sm mt-1">
                              {formErrors[contact.id].firstName}
                            </p>
                          )}
                        </div>

                        <div>
                          <Label htmlFor={`${contact.id}-lastName`}>
                            Last Name *
                          </Label>
                          <Input
                            id={`${contact.id}-lastName`}
                            value={contact.lastName}
                            onChange={(e) =>
                              updateContact(
                                contact.id,
                                "lastName",
                                e.target.value,
                              )
                            }
                            className={
                              formErrors[contact.id]?.lastName
                                ? "border-red-500"
                                : ""
                            }
                            placeholder="Last name"
                          />
                          {formErrors[contact.id]?.lastName && (
                            <p className="text-red-500 text-sm mt-1">
                              {formErrors[contact.id].lastName}
                            </p>
                          )}
                        </div>

                        <div>
                          <Label htmlFor={`${contact.id}-relationship`}>
                            Relationship *
                          </Label>
                          <Select
                            value={contact.relationship}
                            onValueChange={(value) =>
                              updateContact(contact.id, "relationship", value)
                            }
                          >
                            <SelectTrigger
                              className={
                                formErrors[contact.id]?.relationship
                                  ? "border-red-500"
                                  : ""
                              }
                            >
                              <SelectValue placeholder="Select relationship" />
                            </SelectTrigger>
                            <SelectContent>
                              {relationshipOptions.map((option) => (
                                <SelectItem key={option} value={option}>
                                  {option}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          {formErrors[contact.id]?.relationship && (
                            <p className="text-red-500 text-sm mt-1">
                              {formErrors[contact.id].relationship}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Contact Information */}
                    <div>
                      <h4 className="font-medium text-healthcare-text mb-3">
                        Contact Information
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor={`${contact.id}-primaryPhone`}>
                            Primary Phone *
                          </Label>
                          <div className="relative">
                            <Phone className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                            <Input
                              id={`${contact.id}-primaryPhone`}
                              value={contact.primaryPhone}
                              onChange={(e) =>
                                handlePhoneInput(
                                  contact.id,
                                  "primaryPhone",
                                  e.target.value,
                                )
                              }
                              className={`pl-10 ${formErrors[contact.id]?.primaryPhone ? "border-red-500" : ""}`}
                              placeholder="9876543210"
                              maxLength={10}
                            />
                          </div>
                          {formErrors[contact.id]?.primaryPhone && (
                            <p className="text-red-500 text-sm mt-1">
                              {formErrors[contact.id].primaryPhone}
                            </p>
                          )}
                        </div>

                        <div>
                          <Label htmlFor={`${contact.id}-secondaryPhone`}>
                            Secondary Phone
                          </Label>
                          <div className="relative">
                            <Phone className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                            <Input
                              id={`${contact.id}-secondaryPhone`}
                              value={contact.secondaryPhone || ""}
                              onChange={(e) =>
                                handlePhoneInput(
                                  contact.id,
                                  "secondaryPhone",
                                  e.target.value,
                                )
                              }
                              className="pl-10"
                              placeholder="9876543210"
                              maxLength={10}
                            />
                          </div>
                        </div>

                        <div className="md:col-span-2">
                          <Label htmlFor={`${contact.id}-email`}>
                            Email Address
                          </Label>
                          <div className="relative">
                            <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                            <Input
                              id={`${contact.id}-email`}
                              type="email"
                              value={contact.email || ""}
                              onChange={(e) =>
                                updateContact(
                                  contact.id,
                                  "email",
                                  e.target.value,
                                )
                              }
                              className={`pl-10 ${formErrors[contact.id]?.email ? "border-red-500" : ""}`}
                              placeholder="email@example.com"
                            />
                          </div>
                          {formErrors[contact.id]?.email && (
                            <p className="text-red-500 text-sm mt-1">
                              {formErrors[contact.id].email}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Address */}
                    <div>
                      <h4 className="font-medium text-healthcare-text mb-3">
                        Address
                      </h4>
                      <div className="space-y-3">
                        <div>
                          <Label htmlFor={`${contact.id}-street`}>
                            Street Address
                          </Label>
                          <Input
                            id={`${contact.id}-street`}
                            value={contact.address.street}
                            onChange={(e) =>
                              updateContact(
                                contact.id,
                                "address.street",
                                e.target.value,
                              )
                            }
                            placeholder="123 Main Street"
                          />
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          <div>
                            <Label htmlFor={`${contact.id}-city`}>City</Label>
                            <Input
                              id={`${contact.id}-city`}
                              value={contact.address.city}
                              onChange={(e) =>
                                updateContact(
                                  contact.id,
                                  "address.city",
                                  e.target.value,
                                )
                              }
                              placeholder="City"
                            />
                          </div>

                          <div>
                            <Label htmlFor={`${contact.id}-state`}>State</Label>
                            <Select
                              value={contact.address.state}
                              onValueChange={(value) =>
                                updateContact(
                                  contact.id,
                                  "address.state",
                                  value,
                                )
                              }
                            >
                              <SelectTrigger>
                                <SelectValue placeholder="State" />
                              </SelectTrigger>
                              <SelectContent>
                                {stateOptions.map((state) => (
                                  <SelectItem key={state} value={state}>
                                    {state}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>

                          <div>
                            <Label htmlFor={`${contact.id}-zipCode`}>
                              ZIP Code
                            </Label>
                            <Input
                              id={`${contact.id}-zipCode`}
                              value={contact.address.zipCode}
                              onChange={(e) =>
                                updateContact(
                                  contact.id,
                                  "address.zipCode",
                                  e.target.value,
                                )
                              }
                              placeholder="12345"
                              maxLength={10}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Permissions */}
                    <div>
                      <h4 className="font-medium text-healthcare-text mb-3">
                        Permissions & Authorizations
                      </h4>
                      <div className="space-y-3">
                        <div className="flex items-start space-x-2">
                          <input
                            type="checkbox"
                            id={`${contact.id}-receiveInfo`}
                            checked={contact.isAuthorizedToReceiveInfo}
                            onChange={(e) =>
                              updateContact(
                                contact.id,
                                "isAuthorizedToReceiveInfo",
                                e.target.checked,
                              )
                            }
                            className="mt-1"
                          />
                          <div>
                            <Label
                              htmlFor={`${contact.id}-receiveInfo`}
                              className="cursor-pointer"
                            >
                              Authorized to receive health information
                            </Label>
                            <p className="text-xs text-muted-foreground">
                              This person can receive information about your
                              health status and treatment
                            </p>
                          </div>
                        </div>

                        <div className="flex items-start space-x-2">
                          <input
                            type="checkbox"
                            id={`${contact.id}-makeDecisions`}
                            checked={contact.canMakeHealthcareDecisions}
                            onChange={(e) =>
                              updateContact(
                                contact.id,
                                "canMakeHealthcareDecisions",
                                e.target.checked,
                              )
                            }
                            className="mt-1"
                          />
                          <div>
                            <Label
                              htmlFor={`${contact.id}-makeDecisions`}
                              className="cursor-pointer"
                            >
                              Can make healthcare decisions
                            </Label>
                            <p className="text-xs text-muted-foreground">
                              This person can make healthcare decisions on your
                              behalf if you're unable to
                            </p>
                            {contact.canMakeHealthcareDecisions && (
                              <Badge
                                variant="outline"
                                className="mt-1 text-xs bg-yellow-50 text-yellow-800 border-yellow-300"
                              >
                                Legal documentation may be required
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {/* Submit Button */}
          {contacts.length > 0 && (
            <div className="mt-8 text-center">
              <Button
                onClick={handleSubmit}
                disabled={isSubmitting}
                className="bg-medical-green hover:bg-medical-green/90 text-white px-8 py-3 text-lg font-semibold"
              >
                {isSubmitting ? (
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Saving Contacts...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4" />
                    Save Emergency Contacts
                  </div>
                )}
              </Button>

              <p className="text-sm text-muted-foreground mt-2">
                You can always update your emergency contacts later from your
                patient portal
              </p>
            </div>
          )}
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
              Emergency Contacts Saved!
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Your emergency contact information has been successfully saved.
              These contacts can be reached in case of medical emergencies.
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
