import { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "@/lib/UserContext";
import { generateOnboardingReceiptPDF, PDFData } from "@/lib/pdfGenerator";
import AppHeader from "@/components/AppHeader";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { toast } from "@/hooks/use-toast";
import {
  Heart,
  CheckCircle2,
  Download,
  ArrowRight,
  Clock,
  Phone,
  Mail,
  Calendar,
  Shield,
  FileText,
  Users,
  Activity,
  Star,
  Sparkles,
  Gift,
  AlertCircle,
} from "lucide-react";

interface CompletedStep {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  completedDate: string;
  status: "completed";
}

interface NextStep {
  id: string;
  title: string;
  description: string;
  timeframe: string;
  icon: React.ComponentType<{ className?: string }>;
  priority: "high" | "medium" | "low";
  actionLabel?: string;
  isActionable?: boolean;
  status?: "pending" | "in_progress" | "completed" | "not_started";
}

export default function OnboardingComplete() {
  const navigate = useNavigate();
  const [showPDFDialog, setShowPDFDialog] = useState(false);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);

  // Dialog states
  const [showCallScheduleDialog, setShowCallScheduleDialog] = useState(false);
  const [showNotificationDialog, setShowNotificationDialog] = useState(false);
  const [showWellnessDialog, setShowWellnessDialog] = useState(false);

  // Loading states
  const [isSchedulingCall, setIsSchedulingCall] = useState(false);
  const [isSavingNotifications, setIsSavingNotifications] = useState(false);
  const [isEnrollingWellness, setIsEnrollingWellness] = useState(false);
  const [isSendingImmediate, setIsSendingImmediate] = useState(false);
  const [deliveryStatus, setDeliveryStatus] = useState<{
    [key: string]: { success: boolean; message: string; timestamp: Date };
  }>({});

  // Local form states for dialogs
  const [localTimeSlot, setLocalTimeSlot] = useState<
    "morning" | "afternoon" | "evening" | "anytime" | null
  >(null);
  const [localDeliveryMethods, setLocalDeliveryMethods] = useState<
    ("email" | "sms")[]
  >(["email"]);
  const [localAppointmentReminders, setLocalAppointmentReminders] =
    useState(true);
  const [localTestResults, setLocalTestResults] = useState(true);
  const [localHealthTips, setLocalHealthTips] = useState(true);
  const [localWellnessPlan, setLocalWellnessPlan] = useState<
    "basic" | "premium" | null
  >(null);

  const {
    userData,
    onboardingProgress,
    appointments,
    healthcareTasks,
    updateHealthcareTask,
    updateHealthcareTasksBatch,
  } = useUser();

  // Initialize healthcare tasks as completed when reaching onboarding complete
  useEffect(() => {
    // Only run once when component mounts
    // Mark document verification as completed since onboarding is complete
    if (healthcareTasks.documentVerificationStatus === "pending") {
      updateHealthcareTask("documentVerificationStatus", "completed");
    }
  }, []); // Empty dependency array to run only once

  // Update local form states when healthcare tasks change
  useEffect(() => {
    setLocalTimeSlot(healthcareTasks.selectedTimeSlot);
    setLocalDeliveryMethods([...healthcareTasks.selectedDeliveryMethods]);
    setLocalAppointmentReminders(healthcareTasks.appointmentReminders);
    setLocalTestResults(healthcareTasks.testResults);
    setLocalHealthTips(healthcareTasks.healthTips);
    setLocalWellnessPlan(healthcareTasks.selectedWellnessPlan);
  }, [healthcareTasks]);

  const patientName = userData
    ? `${userData.firstName} ${userData.lastName}`
    : "Patient";
  const firstName = userData?.firstName || "Patient";

  const completedSteps: CompletedStep[] = [
    {
      id: "registration",
      title: "Account Registration",
      description: "Personal details and account security setup",
      icon: Users,
      completedDate: "December 15, 2024",
      status: "completed",
    },
    {
      id: "documents",
      title: "Document Upload",
      description: "ID verification and insurance card upload",
      icon: FileText,
      completedDate: "December 15, 2024",
      status: "completed",
    },
    {
      id: "appointment",
      title: "First Appointment Scheduled",
      description: "Consultation with Dr. Sarah Smith on Jan 30, 2025",
      icon: Calendar,
      completedDate: "December 15, 2024",
      status: "completed",
    },
    {
      id: "consent-forms",
      title: "Consent Forms Signed",
      description: "HIPAA agreement and treatment consent completed",
      icon: Shield,
      completedDate: "December 15, 2024",
      status: "completed",
    },
    {
      id: "medical-history",
      title: "Medical History Complete",
      description: "Health background and family history provided",
      icon: Activity,
      completedDate: "December 15, 2024",
      status: "completed",
    },
    {
      id: "emergency-contacts",
      title: "Emergency Contacts Added",
      description: "Emergency contact information saved",
      icon: Phone,
      completedDate: "December 15, 2024",
      status: "completed",
    },
  ];

  const nextSteps: NextStep[] = useMemo(
    () => [
      {
        id: "verification",
        title: "Document Verification",
        description:
          "Track the status of your document verification process. View detailed status updates and receive notifications.",
        timeframe:
          healthcareTasks.documentVerificationStatus === "completed"
            ? "Completed"
            : healthcareTasks.documentVerificationStatus === "in_progress"
              ? "In Progress"
              : "Within 24 hours",
        icon: Shield,
        priority: "high",
        actionLabel: "View Status",
        isActionable: true,
        status: healthcareTasks.documentVerificationStatus,
      },
      {
        id: "clinic-contact",
        title: "Clinic Confirmation Call",
        description:
          "Schedule your preferred time for our team to call you and confirm your appointment details.",
        timeframe: healthcareTasks.callScheduled
          ? `Scheduled (${healthcareTasks.selectedTimeSlot || "Time TBD"})`
          : "Schedule Now",
        icon: Phone,
        priority: "high",
        actionLabel: healthcareTasks.callScheduled
          ? "Reschedule Call"
          : "Schedule Call",
        isActionable: true,
        status: healthcareTasks.callScheduled ? "completed" : "not_started",
      },
      {
        id: "appointment-reminder",
        title: "Notification Preferences",
        description:
          "Set up your notification preferences for appointment reminders, test results, and health updates.",
        timeframe: healthcareTasks.notificationsConfigured
          ? `Active (${healthcareTasks.selectedDeliveryMethods.join(" + ")})`
          : "Set Up Now",
        icon: Calendar,
        priority: "medium",
        actionLabel: healthcareTasks.notificationsConfigured
          ? "Reconfigure"
          : "Configure",
        isActionable: true,
        status: healthcareTasks.notificationsConfigured
          ? "completed"
          : "not_started",
      },
      {
        id: "portal-access",
        title: "Patient Portal Access",
        description:
          "Access your complete healthcare dashboard with test results, messaging, prescription management, and appointment scheduling.",
        timeframe: "Available now",
        icon: Activity,
        priority: "medium",
        actionLabel: "Access Portal",
        isActionable: true,
        status: "completed",
      },
      {
        id: "wellness-program",
        title: "Wellness Program Enrollment",
        description:
          "Join our comprehensive wellness program with health tracking, personalized recommendations, and lifestyle coaching.",
        timeframe: healthcareTasks.wellnessEnrolled
          ? `Enrolled (${healthcareTasks.selectedWellnessPlan || "Plan TBD"})`
          : "Enroll Now",
        icon: Star,
        priority: "low",
        actionLabel: healthcareTasks.wellnessEnrolled
          ? "Change Plan"
          : "Enroll",
        isActionable: true,
        status: healthcareTasks.wellnessEnrolled ? "completed" : "not_started",
      },
    ],
    [healthcareTasks],
  );

  const handleDownloadPDF = () => {
    if (!userData) return;

    setIsGeneratingPDF(true);

    // Generate actual PDF with real data
    setTimeout(() => {
      const pdfData: PDFData = {
        userData,
        onboardingProgress,
        appointments,
        completionDate: new Date().toLocaleDateString(),
      };

      try {
        generateOnboardingReceiptPDF(pdfData);
        setIsGeneratingPDF(false);
        setShowPDFDialog(false);
      } catch (error) {
        console.error("PDF generation failed:", error);
        setIsGeneratingPDF(false);
        console.error("PDF generation failed");
        setDeliveryStatus((prev) => ({
          ...prev,
          pdf_error: {
            success: false,
            message: "PDF generation failed. Please try again.",
            timestamp: new Date(),
          },
        }));
      }
    }, 1500);
  };

  const getPriorityColor = (priority: NextStep["priority"]) => {
    switch (priority) {
      case "high":
        return "bg-red-100 text-red-800";
      case "medium":
        return "bg-blue-100 text-blue-800";
      case "low":
        return "bg-green-100 text-green-800";
    }
  };

  const getPriorityIcon = (priority: NextStep["priority"]) => {
    switch (priority) {
      case "high":
        return "";
      case "medium":
        return "";
      case "low":
        return "";
    }
  };

  const getStatusColor = (status: NextStep["status"]) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800";
      case "in_progress":
        return "bg-yellow-100 text-yellow-800";
      case "pending":
        return "bg-gray-100 text-gray-800";
      case "not_started":
      default:
        return ""; // No status badge for not_started
    }
  };

  const handleStepAction = (stepId: string) => {
    switch (stepId) {
      case "verification":
        if (healthcareTasks.documentVerificationStatus === "pending") {
          updateHealthcareTask("documentVerificationStatus", "in_progress");
          // Simulate document verification process
          setTimeout(() => {
            updateHealthcareTask("documentVerificationStatus", "completed");
          }, 3000);
        }
        break;
      case "clinic-contact":
        setLocalTimeSlot(healthcareTasks.selectedTimeSlot || null);
        setShowCallScheduleDialog(true);
        break;
      case "appointment-reminder":
        setLocalDeliveryMethods([...healthcareTasks.selectedDeliveryMethods]);
        setLocalAppointmentReminders(healthcareTasks.appointmentReminders);
        setLocalTestResults(healthcareTasks.testResults);
        setLocalHealthTips(healthcareTasks.healthTips);
        setShowNotificationDialog(true);
        break;
      case "portal-access":
        navigate("/dashboard");
        break;
      case "wellness-program":
        setLocalWellnessPlan(healthcareTasks.selectedWellnessPlan || null);
        setShowWellnessDialog(true);
        break;
    }
  };

  const handleScheduleCall = async () => {
    if (!localTimeSlot) {
      setDeliveryStatus((prev) => ({
        ...prev,
        call_error: {
          success: false,
          message: "Please select a preferred time slot",
          timestamp: new Date(),
        },
      }));
      return;
    }

    setIsSchedulingCall(true);

    // Simulate call scheduling (no actual API call)
    setTimeout(() => {
      updateHealthcareTasksBatch({
        callScheduled: true,
        selectedTimeSlot: localTimeSlot,
      });
      setShowCallScheduleDialog(false);
      setDeliveryStatus((prev) => ({
        ...prev,
        call_success: {
          success: true,
          message: `Call preference saved for ${localTimeSlot} time slot`,
          timestamp: new Date(),
        },
      }));
      setIsSchedulingCall(false);

      // Show success toast
      toast({
        title: "Call Scheduled!",
        description: `Your preferred time slot (${localTimeSlot}) has been saved successfully.`,
      });
    }, 1000);
  };

  const handleSaveNotifications = async () => {
    if (localDeliveryMethods.length === 0) {
      setDeliveryStatus((prev) => ({
        ...prev,
        notification_error: {
          success: false,
          message: "Please select at least one delivery method",
          timestamp: new Date(),
        },
      }));
      return;
    }

    setIsSavingNotifications(true);

    // Simulate saving (like other handlers)
    setTimeout(() => {
      updateHealthcareTasksBatch({
        notificationsConfigured: true,
        selectedDeliveryMethods: localDeliveryMethods,
        appointmentReminders: localAppointmentReminders,
        testResults: localTestResults,
        healthTips: localHealthTips,
      });
      setShowNotificationDialog(false);
      setDeliveryStatus((prev) => ({
        ...prev,
        notification_success: {
          success: true,
          message: "Notification preferences saved successfully!",
          timestamp: new Date(),
        },
      }));
      setIsSavingNotifications(false);

      // Show success toast
      toast({
        title: "Preferences Saved!",
        description: `Your notification preferences have been configured successfully.`,
      });
    }, 1000);
  };

  const handleWellnessEnrollment = async () => {
    if (!localWellnessPlan) {
      setDeliveryStatus((prev) => ({
        ...prev,
        wellness_error: {
          success: false,
          message: "Please select a wellness plan",
          timestamp: new Date(),
        },
      }));
      return;
    }

    setIsEnrollingWellness(true);
    try {
      // Simulate enrollment API call
      await new Promise((resolve) => setTimeout(resolve, 2000));

      updateHealthcareTasksBatch({
        wellnessEnrolled: true,
        selectedWellnessPlan: localWellnessPlan,
      });
      setShowWellnessDialog(false);
      setDeliveryStatus((prev) => ({
        ...prev,
        wellness_success: {
          success: true,
          message: `Successfully enrolled in ${localWellnessPlan} wellness plan!`,
          timestamp: new Date(),
        },
      }));

      // Show success toast
      toast({
        title: "Wellness Enrollment Complete!",
        description: `You've been successfully enrolled in the ${localWellnessPlan} wellness plan.`,
      });
    } catch (error) {
      console.error("Error enrolling in wellness program:", error);
      setDeliveryStatus((prev) => ({
        ...prev,
        wellness_error: {
          success: false,
          message: "Failed to enroll in wellness program. Please try again.",
          timestamp: new Date(),
        },
      }));
    } finally {
      setIsEnrollingWellness(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light-blue via-background to-medical-light-green relative overflow-hidden">
      <AppHeader />

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {/* Celebration Header */}
          <div className="text-center mb-12">
            <div className="relative inline-block mb-6">
              <div className="bg-green-100 w-24 h-24 rounded-full flex items-center justify-center mx-auto relative z-10">
                <CheckCircle2 className="h-12 w-12 text-green-600" />
              </div>
              <div className="absolute inset-0 animate-ping bg-green-200 rounded-full opacity-75"></div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-center gap-2 text-green-600 font-medium">
                <Gift className="h-5 w-5" />
                <span>Congratulations!</span>
              </div>

              <h1 className="text-4xl md:text-5xl font-bold text-healthcare-text">
                Onboarding Complete!
              </h1>

              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Welcome to HealthCare Plus, {firstName}! You've successfully
                completed all onboarding steps and are ready to begin your
                healthcare journey with us.
              </p>

              <div className="flex items-center justify-center gap-2 bg-medical-light-green/20 px-4 py-2 rounded-full text-medical-green font-medium">
                <Star className="h-4 w-4" />
                <span>All steps completed successfully</span>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 mb-12">
            {/* Completed Steps Summary */}
            <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-2xl text-healthcare-text flex items-center gap-2">
                  <CheckCircle2 className="h-6 w-6 text-green-600" />
                  Completed Steps
                </CardTitle>
                <CardDescription className="text-base">
                  Here's a summary of everything you've accomplished
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-4">
                {completedSteps.map((step, index) => (
                  <div
                    key={step.id}
                    className="flex items-start gap-4 p-4 bg-green-50/50 rounded-lg border border-green-200"
                  >
                    <div className="bg-green-100 p-2 rounded-full flex-shrink-0">
                      <step.icon className="h-5 w-5 text-green-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-healthcare-text">
                          {step.title}
                        </h3>
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      </div>
                      <p className="text-sm text-muted-foreground mb-1">
                        {step.description}
                      </p>
                      <p className="text-xs text-green-600 font-medium">
                        Completed on {step.completedDate}
                      </p>
                    </div>
                  </div>
                ))}

                <div className="pt-4 border-t border-green-200">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-healthcare-text">
                      Total Progress
                    </span>
                    <Badge className="bg-green-100 text-green-800">
                      6/6 Steps Complete
                    </Badge>
                  </div>
                  <div className="mt-2 bg-green-200 rounded-full h-2">
                    <div
                      className="bg-green-600 h-2 rounded-full transition-all duration-1000"
                      style={{ width: "100%" }}
                    ></div>
                  </div>
                  <p className="text-xs text-green-600 font-medium mt-1">
                    100% Complete!
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Next Steps */}
            <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-2xl text-healthcare-text flex items-center gap-2">
                  <Clock className="h-6 w-6 text-medical-blue" />
                  What Happens Next
                </CardTitle>
                <CardDescription className="text-base">
                  Here's what you can expect in the coming days
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-4">
                {nextSteps.map((step, index) => (
                  <div
                    key={step.id}
                    className="flex items-start gap-4 p-4 bg-blue-50/50 rounded-lg border border-blue-200 hover:shadow-md transition-all duration-200"
                  >
                    <div className="bg-medical-blue/10 p-2 rounded-full flex-shrink-0">
                      <step.icon className="h-5 w-5 text-medical-blue" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-healthcare-text">
                          {step.title}
                        </h3>
                        <div className="flex gap-1">
                          <Badge
                            className={`text-xs ${getPriorityColor(step.priority)}`}
                          >
                            {step.priority}
                          </Badge>
                          {step.status && step.status !== "not_started" && (
                            <Badge
                              className={`text-xs ${getStatusColor(step.status)}`}
                            >
                              {step.status === "in_progress"
                                ? "In Progress"
                                : step.status === "completed"
                                  ? "Completed"
                                  : "Pending"}
                            </Badge>
                          )}
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground mb-3">
                        {step.description}
                      </p>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1 text-xs text-medical-blue font-medium">
                          <Clock className="h-3 w-3" />
                          <span>{step.timeframe}</span>
                        </div>
                        {step.isActionable && (
                          <Button
                            size="sm"
                            variant={
                              step.status === "completed"
                                ? "outline"
                                : "default"
                            }
                            onClick={() => handleStepAction(step.id)}
                            className="h-8 px-3 text-xs"
                          >
                            {step.actionLabel}
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Action Cards */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
            {/* Download Forms */}
            <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
              <CardHeader className="text-center">
                <div className="bg-medical-blue/10 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-2">
                  <Download className="h-6 w-6 text-medical-blue" />
                </div>
                <CardTitle className="text-lg text-healthcare-text">
                  Download Forms
                </CardTitle>
                <CardDescription>
                  Get a PDF copy of all your submitted forms and information
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button
                  onClick={() => setShowPDFDialog(true)}
                  className="w-full bg-medical-blue hover:bg-medical-blue/90"
                >
                  <Download className="mr-2 h-4 w-4" />
                  Download PDF
                </Button>
              </CardContent>
            </Card>

            {/* Patient Portal */}
            <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
              <CardHeader className="text-center">
                <div className="bg-medical-green/10 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-2">
                  <Activity className="h-6 w-6 text-medical-green" />
                </div>
                <CardTitle className="text-lg text-healthcare-text">
                  Patient Portal
                </CardTitle>
                <CardDescription>
                  Access your complete healthcare dashboard and records
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button
                  onClick={() => navigate("/dashboard")}
                  className="w-full bg-medical-green hover:bg-medical-green/90"
                >
                  <ArrowRight className="mr-2 h-4 w-4" />
                  Access Portal
                </Button>
              </CardContent>
            </Card>

            {/* Contact Support */}
            <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
              <CardHeader className="text-center">
                <div className="bg-orange-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-2">
                  <Phone className="h-6 w-6 text-orange-600" />
                </div>
                <CardTitle className="text-lg text-healthcare-text">
                  Need Help?
                </CardTitle>
                <CardDescription>
                  Our support team is here to assist you with any questions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center gap-2">
                    <Phone className="h-4 w-4 text-muted-foreground" />
                    <span>98765 43210</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Mail className="h-4 w-4 text-muted-foreground" />
                    <span>support@healthcareplus.com</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Welcome Message */}
          <Card className="border-0 shadow-xl bg-gradient-to-r from-medical-light-blue to-medical-light-green/50">
            <CardContent className="p-8 text-center">
              <h2 className="text-2xl font-bold text-healthcare-text mb-4">
                Welcome to Your HealthCare Journey!
              </h2>
              <p className="text-lg text-muted-foreground mb-6 max-w-3xl mx-auto">
                Thank you for choosing HealthCare Plus. We're committed to
                providing you with exceptional, personalized healthcare
                services. Our team is excited to work with you on your health
                and wellness goals.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  onClick={() => navigate("/dashboard")}
                  size="lg"
                  className="bg-medical-blue hover:bg-medical-blue/90 text-white px-8"
                >
                  <Star className="mr-2 h-5 w-5" />
                  Start Using Your Portal
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  onClick={() => navigate("/")}
                  className="border-medical-green text-medical-green hover:bg-medical-green hover:text-white px-8"
                >
                  Return to Homepage
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* PDF Download Dialog */}
      <Dialog open={showPDFDialog} onOpenChange={setShowPDFDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader className="text-center">
            <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Download className="h-8 w-8 text-blue-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              Download Your Forms
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Generate a PDF copy of all your submitted onboarding forms and
              information for your records.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-medium text-healthcare-text mb-2">
                Your PDF will include:
              </h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Personal information and contact details</li>
                <li>• Signed consent forms and agreements</li>
                <li>• Medical history and health information</li>
                <li>• Emergency contact information</li>
                <li>• Appointment scheduling details</li>
              </ul>
            </div>

            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={() => setShowPDFDialog(false)}
                className="flex-1"
                disabled={isGeneratingPDF}
              >
                Cancel
              </Button>
              <Button
                onClick={handleDownloadPDF}
                disabled={isGeneratingPDF}
                className="flex-1 bg-medical-blue hover:bg-medical-blue/90"
              >
                {isGeneratingPDF ? (
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Generating...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <Download className="h-4 w-4" />
                    Generate PDF
                  </div>
                )}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Call Schedule Dialog */}
      <Dialog
        open={showCallScheduleDialog}
        onOpenChange={setShowCallScheduleDialog}
      >
        <DialogContent className="sm:max-w-lg max-h-[90vh] overflow-y-auto">
          <DialogHeader className="text-center">
            <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Phone className="h-8 w-8 text-green-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              Schedule Confirmation Call
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Choose your preferred time for our team to call and confirm your
              appointment details.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6">
            <div className="space-y-3">
              <label className="text-sm font-medium text-healthcare-text">
                Preferred Call Time *
              </label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { value: "morning", label: "Morning", subtitle: "9-11 AM" },
                  {
                    value: "afternoon",
                    label: "Afternoon",
                    subtitle: "1-3 PM",
                  },
                  { value: "evening", label: "Evening", subtitle: "5-7 PM" },
                  { value: "anytime", label: "Any Time", subtitle: "Flexible" },
                ].map((slot) => (
                  <Button
                    key={slot.value}
                    variant={
                      localTimeSlot === slot.value ? "default" : "outline"
                    }
                    className={`h-auto p-3 justify-start flex-col items-start ${
                      localTimeSlot === slot.value
                        ? "bg-medical-green text-white"
                        : "hover:bg-green-50"
                    }`}
                    onClick={() => setLocalTimeSlot(slot.value as any)}
                  >
                    <span className="font-medium">{slot.label}</span>
                    <span className="text-xs opacity-70">{slot.subtitle}</span>
                  </Button>
                ))}
              </div>
              {localTimeSlot && (
                <div className="text-sm text-green-600 font-medium">
                  ✓{" "}
                  {localTimeSlot === "morning"
                    ? "Morning (9-11 AM)"
                    : localTimeSlot === "afternoon"
                      ? "Afternoon (1-3 PM)"
                      : localTimeSlot === "evening"
                        ? "Evening (5-7 PM)"
                        : "Any Time"}{" "}
                  selected
                </div>
              )}
            </div>

            <div className="bg-green-50 p-4 rounded-lg">
              <h4 className="font-medium text-healthcare-text mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                Your call will include:
              </h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Confirm appointment date and time</li>
                <li>• Review your medical history</li>
                <li>• Answer any questions you may have</li>
                <li>• Discuss preparation instructions</li>
                <li>• Verify contact information</li>
              </ul>
            </div>

            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="flex items-start gap-2">
                <Phone className="h-4 w-4 text-blue-600 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-blue-900">
                    We'll call: {userData?.phone || "Phone number not provided"}
                  </p>
                  <p className="text-blue-700">
                    Make sure this number is correct and available
                  </p>
                </div>
              </div>
            </div>

            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={() => {
                  setShowCallScheduleDialog(false);
                  setLocalTimeSlot(healthcareTasks.selectedTimeSlot);
                }}
                className="flex-1"
                disabled={isSchedulingCall}
              >
                Cancel
              </Button>
              <Button
                onClick={handleScheduleCall}
                disabled={!localTimeSlot || isSchedulingCall}
                className="flex-1 bg-medical-green hover:bg-medical-green/90"
              >
                {isSchedulingCall ? (
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Scheduling...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <Phone className="h-4 w-4" />
                    Schedule Call
                  </div>
                )}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Notification Settings Dialog */}
      <Dialog
        open={showNotificationDialog}
        onOpenChange={setShowNotificationDialog}
      >
        <DialogContent className="sm:max-w-lg max-h-[90vh] overflow-y-auto">
          <DialogHeader className="text-center">
            <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Calendar className="h-8 w-8 text-blue-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              Notification Preferences
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Set your preferences for receiving important health updates and
              reminders.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6">
            <div className="space-y-3">
              <label className="text-sm font-medium text-healthcare-text">
                Notification Types
              </label>

              <div className="space-y-2">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                  <div>
                    <p className="font-medium text-sm">Appointment Reminders</p>
                    <p className="text-xs text-muted-foreground">
                      24 hours before appointments
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={localAppointmentReminders}
                      onChange={(e) =>
                        setLocalAppointmentReminders(e.target.checked)
                      }
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                  <div>
                    <p className="font-medium text-sm">Test Results</p>
                    <p className="text-xs text-muted-foreground">
                      When lab results are available
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={localTestResults}
                      onChange={(e) => setLocalTestResults(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                  <div>
                    <p className="font-medium text-sm">Health Tips</p>
                    <p className="text-xs text-muted-foreground">
                      Weekly wellness recommendations
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={localHealthTips}
                      onChange={(e) => setLocalHealthTips(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <label className="text-sm font-medium text-healthcare-text">
                Delivery Method *
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { value: "email", label: "Email Only", icon: Mail },
                  { value: "sms", label: "SMS Only", icon: Phone },
                  { value: "both", label: "Both", icon: Sparkles },
                ].map((method) => {
                  const isSelected =
                    method.value === "both"
                      ? localDeliveryMethods.includes("email") &&
                        localDeliveryMethods.includes("sms")
                      : localDeliveryMethods.includes(
                          method.value as "email" | "sms",
                        );

                  return (
                    <Button
                      key={method.value}
                      variant={isSelected ? "default" : "outline"}
                      size="sm"
                      className={`h-auto p-3 flex-col gap-1 ${
                        isSelected
                          ? "bg-medical-blue text-white"
                          : "hover:bg-blue-50"
                      }`}
                      onClick={async () => {
                        if (isSendingImmediate) return;

                        const oldMethods = [...localDeliveryMethods];
                        let newMethods: ("email" | "sms")[];

                        if (method.value === "both") {
                          newMethods = ["email", "sms"];
                        } else if (method.value === "email") {
                          newMethods = ["email"];
                        } else {
                          newMethods = ["sms"];
                        }

                        setLocalDeliveryMethods(newMethods);

                        // Just track selection - no actual sending
                        const newlySelectedMethods = newMethods.filter(
                          (m) => !oldMethods.includes(m),
                        );

                        if (newlySelectedMethods.length > 0) {
                          for (const newMethod of newlySelectedMethods) {
                            const statusKey = `selected_${newMethod}`;
                            setDeliveryStatus((prev) => ({
                              ...prev,
                              [statusKey]: {
                                success: true,
                                message: `${newMethod === "email" ? "Email" : "SMS"} notifications selected`,
                                timestamp: new Date(),
                              },
                            }));
                          }
                        }
                      }}
                      disabled={isSendingImmediate}
                    >
                      <method.icon className="h-4 w-4" />
                      <span className="text-xs">{method.label}</span>
                    </Button>
                  );
                })}
              </div>
              <div className="text-sm text-muted-foreground">
                Selected:{" "}
                {localDeliveryMethods.includes("email") &&
                localDeliveryMethods.includes("sms")
                  ? "Email + SMS"
                  : localDeliveryMethods.includes("email")
                    ? "Email only"
                    : "SMS only"}
                {isSendingImmediate && (
                  <span className="text-blue-600 font-medium ml-2">
                    <div className="inline-flex items-center gap-1">
                      <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600"></div>
                      Sending welcome message...
                    </div>
                  </span>
                )}
              </div>
            </div>

            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="flex items-start gap-2 mb-3">
                <Shield className="h-4 w-4 text-blue-600 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-blue-900">
                    Contact Information
                  </p>
                  <p className="text-blue-700">
                    Email: {userData?.email || "Not provided"}
                  </p>
                  <p className="text-blue-700">
                    Phone: {userData?.phone || "Not provided"}
                  </p>
                </div>
              </div>
            </div>

            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={() => {
                  setShowNotificationDialog(false);
                  // Reset to defaults
                  setLocalDeliveryMethods(["email"]);
                  setLocalAppointmentReminders(true);
                  setLocalTestResults(true);
                  setLocalHealthTips(true);
                }}
                className="flex-1"
                disabled={isSavingNotifications}
              >
                Cancel
              </Button>
              <Button
                onClick={handleSaveNotifications}
                disabled={
                  localDeliveryMethods.length === 0 || isSavingNotifications
                }
                className="flex-1 bg-medical-blue hover:bg-medical-blue/90"
              >
                {isSavingNotifications ? (
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Saving...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <Calendar className="h-4 w-4" />
                    Save Preferences
                  </div>
                )}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Wellness Program Dialog */}
      <Dialog open={showWellnessDialog} onOpenChange={setShowWellnessDialog}>
        <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader className="text-center">
            <div className="bg-gradient-to-br from-yellow-100 to-orange-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Star className="h-8 w-8 text-yellow-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              Wellness Program Enrollment
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Join our comprehensive wellness program for personalized health
              tracking and recommendations.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6">
            <div className="bg-gradient-to-r from-yellow-50 to-orange-50 p-4 rounded-lg border border-yellow-200">
              <h4 className="font-medium text-healthcare-text mb-3 flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-yellow-600" />
                Program Benefits:
              </h4>
              <div className="grid md:grid-cols-2 gap-3">
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3 w-3 text-green-600" />{" "}
                    Personalized health dashboard
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3 w-3 text-green-600" /> Monthly
                    wellness assessments
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3 w-3 text-green-600" />{" "}
                    Nutrition and exercise tracking
                  </li>
                </ul>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3 w-3 text-green-600" /> Health
                    coaching sessions
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3 w-3 text-green-600" />{" "}
                    Preventive care reminders
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-3 w-3 text-green-600" /> Wellness
                    challenges and rewards
                  </li>
                </ul>
              </div>
            </div>

            <div className="space-y-3">
              <label className="text-sm font-medium text-healthcare-text">
                Choose Your Plan *
              </label>

              <div className="grid md:grid-cols-2 gap-4">
                {/* Basic Plan */}
                <div
                  className={`relative p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                    localWellnessPlan === "basic"
                      ? "border-green-500 bg-green-50 shadow-md"
                      : "border-gray-200 bg-white hover:border-green-300 hover:shadow-sm"
                  }`}
                  onClick={() => setLocalWellnessPlan("basic")}
                >
                  {localWellnessPlan === "basic" && (
                    <div className="absolute -top-2 -right-2 bg-green-500 text-white rounded-full p-1">
                      <CheckCircle2 className="h-4 w-4" />
                    </div>
                  )}
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-healthcare-text">
                      Basic Plan
                    </h3>
                    <Badge className="bg-green-100 text-green-800 font-bold">
                      FREE
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Perfect for getting started with health tracking
                  </p>
                  <ul className="text-sm space-y-1">
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-3 w-3 text-green-600" /> Health
                      dashboard
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-3 w-3 text-green-600" /> Basic
                      tracking tools
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-3 w-3 text-green-600" /> Weekly
                      health tips
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-3 w-3 text-green-600" />{" "}
                      Progress reports
                    </li>
                  </ul>
                </div>

                {/* Premium Plan */}
                <div
                  className={`relative p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                    localWellnessPlan === "premium"
                      ? "border-blue-500 bg-blue-50 shadow-md"
                      : "border-gray-200 bg-white hover:border-blue-300 hover:shadow-sm"
                  }`}
                  onClick={() => setLocalWellnessPlan("premium")}
                >
                  {localWellnessPlan === "premium" && (
                    <div className="absolute -top-2 -right-2 bg-blue-500 text-white rounded-full p-1">
                      <CheckCircle2 className="h-4 w-4" />
                    </div>
                  )}
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-healthcare-text flex items-center gap-1">
                      Premium Plan
                      <Sparkles className="h-4 w-4 text-yellow-500" />
                    </h3>
                    <Badge className="bg-blue-100 text-blue-800 font-bold">
                      $29/mo
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Complete wellness support with personal coaching
                  </p>
                  <ul className="text-sm space-y-1">
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-3 w-3 text-blue-600" />{" "}
                      Everything in Basic
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-3 w-3 text-blue-600" />{" "}
                      Personal health coach
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-3 w-3 text-blue-600" /> Custom
                      meal plans
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-3 w-3 text-blue-600" /> 24/7
                      chat support
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-3 w-3 text-blue-600" /> Premium
                      challenges
                    </li>
                  </ul>
                </div>
              </div>

              {localWellnessPlan && (
                <div className="text-sm font-medium text-center p-2 bg-gray-50 rounded">
                  ✓{" "}
                  {localWellnessPlan === "basic"
                    ? "Basic Plan (FREE)"
                    : "Premium Plan ($29/month)"}{" "}
                  selected
                </div>
              )}
            </div>

            {localWellnessPlan === "premium" && (
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <h4 className="font-medium text-blue-900 mb-2 flex items-center gap-2">
                  <Gift className="h-4 w-4" />
                  Special Offer!
                </h4>
                <p className="text-sm text-blue-800">
                  Get your first month free! Premium plan will start at
                  $29/month after your 30-day trial.
                </p>
              </div>
            )}

            <div className="bg-yellow-50 p-4 rounded-lg">
              <h4 className="font-medium text-healthcare-text mb-2 flex items-center gap-2">
                <Shield className="h-4 w-4 text-yellow-600" />
                What happens next?
              </h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Instant access to your wellness dashboard</li>
                <li>• Initial health assessment questionnaire</li>
                <li>• Personalized goal setting session</li>
                {localWellnessPlan === "premium" && (
                  <li>• Schedule your first coaching call within 48 hours</li>
                )}
              </ul>
            </div>

            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={() => {
                  setShowWellnessDialog(false);
                  setLocalWellnessPlan(null);
                }}
                className="flex-1"
                disabled={isEnrollingWellness}
              >
                Maybe Later
              </Button>
              <Button
                onClick={handleWellnessEnrollment}
                disabled={!localWellnessPlan || isEnrollingWellness}
                className={`flex-1 text-white ${
                  localWellnessPlan === "premium"
                    ? "bg-blue-500 hover:bg-blue-600"
                    : "bg-green-500 hover:bg-green-600"
                }`}
              >
                {isEnrollingWellness ? (
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Enrolling...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <Star className="h-4 w-4" />
                    {localWellnessPlan === "premium"
                      ? "Start Free Trial"
                      : "Enroll Free"}
                  </div>
                )}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
