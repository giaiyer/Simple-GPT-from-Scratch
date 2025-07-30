import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
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
import { useUser } from "@/lib/UserContext";
import { generateOnboardingReceiptPDF, PDFData } from "@/lib/pdfGenerator";
import AppHeader from "@/components/AppHeader";
import {
  Calendar,
  User,
  Download,
  Edit,
  Phone,
  Mail,
  CheckCircle2,
  Clock,
  Star,
  Activity,
  Shield,
  Plus,
  Bell,
  Settings,
  Sparkles,
  ChevronRight,
  HeartHandshake,
  Stethoscope,
  FileText,
  Users,
  MapPin,
} from "lucide-react";

export default function Dashboard() {
  const navigate = useNavigate();
  const {
    userData,
    onboardingProgress,
    getCompletionPercentage,
    appointments,
    loadAppointments,
    isAdmin,
    healthcareTasks,
    updateHealthcareTask,
    updateHealthcareTasksBatch,
  } = useUser();
  const [showPDFDialog, setShowPDFDialog] = useState(false);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);

  // Dialog states
  const [showCallScheduleDialog, setShowCallScheduleDialog] = useState(false);
  const [showNotificationDialog, setShowNotificationDialog] = useState(false);
  const [showWellnessDialog, setShowWellnessDialog] = useState(false);
  const [showAppointmentsDialog, setShowAppointmentsDialog] = useState(false);

  // Loading states
  const [isSchedulingCall, setIsSchedulingCall] = useState(false);
  const [isSavingNotifications, setIsSavingNotifications] = useState(false);
  const [isEnrollingWellness, setIsEnrollingWellness] = useState(false);

  // Local form states for dialogs - sync with current healthcare tasks
  const [localTimeSlot, setLocalTimeSlot] = useState<
    "morning" | "afternoon" | "evening" | "anytime" | null
  >(healthcareTasks.selectedTimeSlot);
  const [localDeliveryMethods, setLocalDeliveryMethods] = useState<
    ("email" | "sms")[]
  >(healthcareTasks.selectedDeliveryMethods);
  const [localAppointmentReminders, setLocalAppointmentReminders] = useState(
    healthcareTasks.appointmentReminders,
  );
  const [localTestResults, setLocalTestResults] = useState(
    healthcareTasks.testResults,
  );
  const [localHealthTips, setLocalHealthTips] = useState(
    healthcareTasks.healthTips,
  );
  const [localWellnessPlan, setLocalWellnessPlan] = useState<
    "basic" | "premium" | null
  >(healthcareTasks.selectedWellnessPlan);

  // Update local states when healthcare tasks change
  useEffect(() => {
    setLocalTimeSlot(healthcareTasks.selectedTimeSlot);
    setLocalDeliveryMethods([...healthcareTasks.selectedDeliveryMethods]);
    setLocalAppointmentReminders(healthcareTasks.appointmentReminders);
    setLocalTestResults(healthcareTasks.testResults);
    setLocalHealthTips(healthcareTasks.healthTips);
    setLocalWellnessPlan(healthcareTasks.selectedWellnessPlan);
  }, [healthcareTasks]);

  // Load appointments from database when component mounts
  useEffect(() => {
    const loadAppointmentsFromDB = async () => {
      try {
        const token = localStorage.getItem("healthcarePlus_token");
        if (!token) return;

        const response = await fetch("/api/medical/appointments", {
          headers: {
            "Authorization": `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          // Update appointments in context
          loadAppointments(data.appointments);
          console.log("Loaded appointments from database:", data.appointments);
        }
      } catch (error) {
        console.error("Failed to load appointments:", error);
      }
    };

    if (userData && userData.email) {
      loadAppointmentsFromDB();
    }
  }, [userData, loadAppointments]);

  const userName = userData
    ? `${userData.firstName} ${userData.lastName}`
    : "Patient";
  const firstName = userData?.firstName || "Patient";
  const completionPercentage = getCompletionPercentage();
  const isOnboardingComplete = completionPercentage === 100;

  // Upcoming appointments
  const upcomingAppointments = appointments.filter(
    (apt) => apt.status === "scheduled" && new Date(apt.date) > new Date(),
  );

  // Next appointment
  const nextAppointment = upcomingAppointments.sort(
    (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime(),
  )[0];

  // Quick actions that are immediately actionable
  const quickActions = [
    {
      id: "book-appointment",
      title: "Book New Appointment",
      description: "Schedule your next consultation",
      icon: Calendar,
      color: "bg-blue-600 hover:bg-blue-700",
      action: () => navigate("/schedule-appointment"),
    },
    {
      id: "view-profile",
      title: "Update Profile",
      description: "Manage your personal information",
      icon: User,
      color: "bg-emerald-600 hover:bg-emerald-700",
      action: () => navigate("/profile-edit"),
    },
    {
      id: "notifications",
      title: "Notifications",
      description: healthcareTasks.notificationsConfigured
        ? "Reconfigure preferences"
        : "Set up preferences",
      icon: Bell,
      color: "bg-orange-600 hover:bg-orange-700",
      action: () => setShowNotificationDialog(true),
    },
    {
      id: "wellness",
      title: "Wellness Program",
      description: healthcareTasks.wellnessEnrolled
        ? "Manage your plan"
        : "Join wellness program",
      icon: HeartHandshake,
      color: "bg-purple-600 hover:bg-purple-700",
      action: () => setShowWellnessDialog(true),
    },
  ];

  // Healthcare services
  const healthcareServices = [
    {
      id: "document-verification",
      title: "Document Verification",
      description:
        healthcareTasks.documentVerificationStatus === "completed"
          ? "Documents verified successfully"
          : healthcareTasks.documentVerificationStatus === "in_progress"
            ? "Verification in progress"
            : "Pending verification",
      icon: Shield,
      status: healthcareTasks.documentVerificationStatus,
      completed: healthcareTasks.documentVerificationStatus === "completed",
    },
    {
      id: "clinic-call",
      title: "Clinic Call",
      description: healthcareTasks.callScheduled
        ? `Scheduled for ${healthcareTasks.selectedTimeSlot}`
        : "Schedule confirmation call",
      icon: Phone,
      status: healthcareTasks.callScheduled ? "completed" : "pending",
      completed: healthcareTasks.callScheduled,
      action: () => setShowCallScheduleDialog(true),
    },
  ];

  const handleDownloadReceipt = () => {
    if (!userData) return;

    setIsGeneratingPDF(true);

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
      }
    }, 1000);
  };

  const handleScheduleCall = async () => {
    if (!localTimeSlot) return;
    setIsSchedulingCall(true);
    setTimeout(() => {
      updateHealthcareTasksBatch({
        callScheduled: true,
        selectedTimeSlot: localTimeSlot,
      });
      setShowCallScheduleDialog(false);
      setIsSchedulingCall(false);
    }, 1000);
  };

  const handleSaveNotifications = async () => {
    if (localDeliveryMethods.length === 0) return;
    setIsSavingNotifications(true);

    setTimeout(() => {
      updateHealthcareTasksBatch({
        notificationsConfigured: true,
        selectedDeliveryMethods: localDeliveryMethods,
        appointmentReminders: localAppointmentReminders,
        testResults: localTestResults,
        healthTips: localHealthTips,
      });
      setShowNotificationDialog(false);
      setIsSavingNotifications(false);
    }, 1000);
  };

  const handleWellnessEnrollment = async () => {
    if (!localWellnessPlan) return;
    setIsEnrollingWellness(true);
    setTimeout(() => {
      updateHealthcareTasksBatch({
        wellnessEnrolled: true,
        selectedWellnessPlan: localWellnessPlan,
      });
      setShowWellnessDialog(false);
      setIsEnrollingWellness(false);
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light-blue via-background to-medical-light-green">
      <AppHeader />

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          {/* Welcome Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-healthcare-text mb-3">
              Welcome back, {firstName}! 
            </h1>
            <p className="text-xl text-muted-foreground">
              Your health dashboard is ready to help you stay on track
            </p>
          </div>

          {/* Main Grid Layout */}
          <div className="grid lg:grid-cols-12 gap-8">
            {/* Left Column - Main Content */}
            <div className="lg:col-span-8 space-y-8">
              {/* Next Appointment Card */}
              {nextAppointment ? (
                <Card className="border-0 shadow-xl bg-gradient-to-r from-blue-50 to-blue-100 border-l-4 border-l-blue-500">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className="bg-blue-100 p-3 rounded-full">
                          <Stethoscope className="h-6 w-6 text-blue-600" />
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-healthcare-text">
                            Upcoming Appointment
                          </h3>
                          <p className="text-muted-foreground">
                            {new Date(
                              nextAppointment.date,
                            ).toLocaleDateString()}{" "}
                            at {nextAppointment.time}
                          </p>
                          <p className="text-sm text-blue-600 font-medium">
                            Dr. {nextAppointment.doctorName} -{" "}
                            {nextAppointment.appointmentType}
                          </p>
                        </div>
                      </div>
                      <Button
                        onClick={() => setShowAppointmentsDialog(true)}
                        variant="outline"
                        className="border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white"
                      >
                        View Details
                        <ChevronRight className="ml-1 h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card className="border-0 shadow-xl bg-gradient-to-r from-green-50 to-emerald-100 border-l-4 border-l-green-500">
                  <CardContent className="p-6">
                    {nextAppointment ? (
                      // Show upcoming appointment details
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="bg-green-100 p-3 rounded-full">
                            <Calendar className="h-6 w-6 text-green-600" />
                          </div>
                          <div>
                            <h3 className="text-lg font-semibold text-healthcare-text">
                              Upcoming Appointment
                            </h3>
                            <div className="space-y-1">
                              <p className="text-sm font-medium text-healthcare-text">
                                {nextAppointment.appointmentType} with {nextAppointment.doctorName}
                              </p>
                              <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                <div className="flex items-center gap-1">
                                  <Calendar className="h-3 w-3" />
                                  {new Date(nextAppointment.date).toLocaleDateString('en-US', {
                                    weekday: 'short',
                                    month: 'short',
                                    day: 'numeric',
                                    year: 'numeric'
                                  })}
                                </div>
                                <div className="flex items-center gap-1">
                                  <Clock className="h-3 w-3" />
                                  {nextAppointment.time}
                                </div>
                                <div className="flex items-center gap-1">
                                  <MapPin className="h-3 w-3" />
                                  {nextAppointment.location}
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        <Button
                          onClick={() => navigate("/schedule-appointment")}
                          className="bg-green-600 hover:bg-green-700 text-white"
                        >
                          <Plus className="mr-2 h-4 w-4" />
                          Book New Appointment
                        </Button>
                      </div>
                    ) : (
                      // Show default booking prompt when no upcoming appointments
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="bg-green-100 p-3 rounded-full">
                            <Calendar className="h-6 w-6 text-green-600" />
                          </div>
                          <div>
                            <h3 className="text-lg font-semibold text-healthcare-text">
                              Ready to Book?
                            </h3>
                            <p className="text-muted-foreground">
                              Schedule your next appointment with our healthcare
                              team
                            </p>
                          </div>
                        </div>
                        <Button
                          onClick={() => navigate("/schedule-appointment")}
                          className="bg-green-600 hover:bg-green-700 text-white"
                        >
                          <Plus className="mr-2 h-4 w-4" />
                          Book New Appointment
                        </Button>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Onboarding Progress - Show when incomplete */}
              {!isOnboardingComplete && (
                <Card className="border-0 shadow-xl bg-gradient-to-r from-amber-50 to-orange-100 border-l-4 border-l-amber-500">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className="bg-amber-100 p-3 rounded-full">
                          <Settings className="h-6 w-6 text-amber-600" />
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-healthcare-text mb-1">
                            Complete Your Onboarding
                          </h3>
                          <p className="text-muted-foreground mb-2">
                            {completionPercentage}% complete - Finish setting up
                            your healthcare profile
                          </p>
                          <div className="w-48 bg-amber-200 rounded-full h-2 mb-2">
                            <div
                              className="bg-amber-600 h-2 rounded-full transition-all duration-500"
                              style={{ width: `${completionPercentage}%` }}
                            ></div>
                          </div>
                          <p className="text-sm text-amber-700">
                            {
                              Object.values(onboardingProgress).filter(Boolean)
                                .length
                            }{" "}
                            of {Object.values(onboardingProgress).length} steps
                            completed
                          </p>
                        </div>
                      </div>
                      <Button
                        onClick={() => navigate("/profile-setup")}
                        className="bg-amber-600 hover:bg-amber-700 text-white"
                      >
                        Continue Setup
                        <ChevronRight className="ml-1 h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Onboarding Complete Celebration - Show when 100% complete */}
              {isOnboardingComplete && (
                <Card className="border-0 shadow-xl bg-gradient-to-r from-green-50 to-emerald-100 border-l-4 border-l-green-500">
                  <CardContent className="p-6">
                    <div className="flex items-center gap-4">
                      <div className="bg-green-100 p-3 rounded-full">
                        <CheckCircle2 className="h-6 w-6 text-green-600" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-healthcare-text mb-1">
                          Onboarding Complete!
                        </h3>
                        <p className="text-muted-foreground">
                          Your healthcare profile is fully set up. You're ready
                          to manage your health journey.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Quick Actions */}
              <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-2xl text-healthcare-text flex items-center gap-2">
                    <Activity className="h-6 w-6 text-medical-blue" />
                    Quick Actions
                  </CardTitle>
                  <CardDescription className="text-base">
                    Common tasks and services at your fingertips
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-2 gap-4">
                    {quickActions.map((action) => (
                      <button
                        key={action.id}
                        onClick={action.action}
                        className={`${action.color} text-white rounded-xl p-6 text-left transition-all duration-200 hover:scale-105 hover:shadow-lg group`}
                      >
                        <div className="flex items-start justify-between mb-4">
                          <action.icon className="h-8 w-8 text-white" />
                          <ChevronRight className="h-5 w-5 text-white/70 group-hover:text-white group-hover:translate-x-1 transition-all" />
                        </div>
                        <h3 className="text-lg font-semibold mb-2">
                          {action.title}
                        </h3>
                        <p className="text-white/80 text-sm">
                          {action.description}
                        </p>
                      </button>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Healthcare Services */}
              <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-2xl text-healthcare-text flex items-center gap-2">
                    <Settings className="h-6 w-6 text-medical-blue" />
                    Healthcare Services
                  </CardTitle>
                  <CardDescription className="text-base">
                    Manage your healthcare preferences and documents
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4">
                    {healthcareServices.map((service) => (
                      <div
                        key={service.id}
                        onClick={service.action}
                        className={`p-4 rounded-xl border-2 transition-all duration-200 ${
                          service.completed
                            ? "bg-green-50 border-green-200 hover:bg-green-100"
                            : "bg-gray-50 border-gray-200 hover:bg-blue-50 hover:border-blue-300"
                        } ${service.action ? "cursor-pointer" : ""}`}
                      >
                        <div className="flex items-start gap-3">
                          <div
                            className={`p-3 rounded-lg ${
                              service.completed ? "bg-green-100" : "bg-gray-100"
                            }`}
                          >
                            <service.icon
                              className={`h-5 w-5 ${
                                service.completed
                                  ? "text-green-600"
                                  : "text-gray-600"
                              }`}
                            />
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-2">
                              <h3 className="font-semibold text-healthcare-text">
                                {service.title}
                              </h3>
                              {service.completed && (
                                <CheckCircle2 className="h-5 w-5 text-green-600" />
                              )}
                            </div>
                            <p className="text-sm text-muted-foreground">
                              {service.description}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right Column - Profile & Support */}
            <div className="lg:col-span-4 space-y-6">
              {/* Profile Card */}
              <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-xl text-healthcare-text flex items-center gap-2">
                      <User className="h-5 w-5" />
                      Your Profile
                    </CardTitle>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => navigate("/profile-edit")}
                      className="border-medical-blue text-medical-blue hover:bg-medical-blue hover:text-white"
                    >
                      <Edit className="mr-2 h-4 w-4" />
                      Edit
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  {userData ? (
                    <div className="space-y-4">
                      <div className="text-center pb-4 border-b">
                        <div className="bg-medical-blue/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-3">
                          <User className="h-8 w-8 text-medical-blue" />
                        </div>
                        <h3 className="font-semibold text-healthcare-text text-lg">
                          {userName}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          {userData.email}
                        </p>
                      </div>
                      <div className="space-y-3">
                        <div>
                          <p className="text-xs font-medium text-muted-foreground mb-1">
                            Phone Number
                          </p>
                          <p className="text-sm text-healthcare-text">
                            {userData.phone}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs font-medium text-muted-foreground mb-1">
                            Date of Birth
                          </p>
                          <p className="text-sm text-healthcare-text">
                            {new Date(
                              userData.dateOfBirth,
                            ).toLocaleDateString()}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs font-medium text-muted-foreground mb-1">
                            Gender
                          </p>
                          <p className="text-sm text-healthcare-text capitalize">
                            {userData.gender}
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <User className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                      <p className="text-muted-foreground mb-4">
                        Complete your profile setup
                      </p>
                      <Button
                        onClick={() => navigate("/register")}
                        className="bg-medical-blue hover:bg-medical-blue/90"
                      >
                        Get Started
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Onboarding Status in Sidebar */}
              {!isOnboardingComplete && (
                <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-xl text-healthcare-text flex items-center gap-2">
                      <Settings className="h-5 w-5 text-amber-600" />
                      Setup Progress
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-amber-600 mb-1">
                          {completionPercentage}%
                        </div>
                        <div className="text-sm text-muted-foreground mb-3">
                          Profile Complete
                        </div>
                        <div className="w-full bg-amber-100 rounded-full h-2 mb-3">
                          <div
                            className="bg-amber-600 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${completionPercentage}%` }}
                          ></div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        {Object.entries(onboardingProgress).map(
                          ([key, completed]) => {
                            const stepNames = {
                              uploadDocuments: "Upload Documents",
                              scheduleAppointment: "Schedule Appointment",
                              consentForms: "Consent Forms",
                              medicalHistory: "Medical History",
                              emergencyContacts: "Emergency Contacts",
                            };

                            return (
                              <div
                                key={key}
                                className="flex items-center gap-2 text-sm"
                              >
                                {completed ? (
                                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                                ) : (
                                  <Clock className="h-4 w-4 text-gray-400" />
                                )}
                                <span
                                  className={
                                    completed
                                      ? "text-green-600"
                                      : "text-gray-500"
                                  }
                                >
                                  {stepNames[key as keyof typeof stepNames]}
                                </span>
                              </div>
                            );
                          },
                        )}
                      </div>

                      <Button
                        onClick={() => navigate("/profile-setup")}
                        className="w-full bg-amber-600 hover:bg-amber-700 text-white"
                        size="sm"
                      >
                        Continue Setup
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Quick Downloads */}
              <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-xl text-healthcare-text flex items-center gap-2">
                    <Download className="h-5 w-5" />
                    Documents
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Button
                    onClick={() => setShowPDFDialog(true)}
                    variant="outline"
                    className="w-full justify-start border-medical-blue text-medical-blue hover:bg-medical-blue hover:text-white"
                  >
                    <Download className="mr-2 h-4 w-4" />
                    Download Receipt
                  </Button>
                </CardContent>
              </Card>

              {/* Support */}
              <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-xl text-healthcare-text flex items-center gap-2">
                    <Phone className="h-5 w-5 text-medical-blue" />
                    Need Help?
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg">
                    <Phone className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium">98765 43210</span>
                  </div>
                  <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg">
                    <Mail className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium">
                      support@healthcareplus.com
                    </span>
                  </div>
                  {isAdmin() && (
                    <Button
                      variant="outline"
                      onClick={() => navigate("/ehr-dashboard")}
                      className="w-full border-purple-600 text-purple-600 hover:bg-purple-600 hover:text-white"
                    >
                      <Settings className="mr-2 h-4 w-4" />
                      Admin Portal
                    </Button>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>

      {/* Appointments View Dialog */}
      <Dialog
        open={showAppointmentsDialog}
        onOpenChange={setShowAppointmentsDialog}
      >
        <DialogContent className="sm:max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader className="text-center">
            <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Calendar className="h-8 w-8 text-blue-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              Your Appointments
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              View and manage all your scheduled appointments
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6">
            {appointments.length > 0 ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-healthcare-text">
                    All Appointments ({appointments.length})
                  </h3>
                  <Button
                    onClick={() => {
                      setShowAppointmentsDialog(false);
                      navigate("/schedule-appointment");
                    }}
                    size="sm"
                    className="bg-medical-blue hover:bg-medical-blue/90"
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    Book New
                  </Button>
                </div>

                <div className="grid gap-4">
                  {appointments
                    .sort(
                      (a, b) =>
                        new Date(b.date).getTime() - new Date(a.date).getTime(),
                    )
                    .map((appointment) => (
                      <Card
                        key={appointment.id}
                        className={`border-l-4 ${
                          appointment.status === "scheduled"
                            ? new Date(appointment.date) > new Date()
                              ? "border-l-blue-500 bg-blue-50/50" // Future appointments
                              : "border-l-orange-500 bg-orange-50/50" // Past scheduled appointments
                            : appointment.status === "completed"
                              ? "border-l-green-500 bg-green-50/50"
                              : "border-l-red-500 bg-red-50/50"
                        }`}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-2">
                                <h4 className="font-semibold text-healthcare-text">
                                  {appointment.appointmentType}
                                </h4>
                                <Badge
                                  className={`text-xs ${
                                    appointment.status === "scheduled"
                                      ? new Date(appointment.date) > new Date()
                                        ? "bg-blue-100 text-blue-800" // Upcoming
                                        : "bg-orange-100 text-orange-800" // Past scheduled
                                      : appointment.status === "completed"
                                        ? "bg-green-100 text-green-800"
                                        : "bg-red-100 text-red-800"
                                  }`}
                                >
                                  {appointment.status === "scheduled" &&
                                  new Date(appointment.date) > new Date()
                                    ? "Upcoming"
                                    : appointment.status}
                                </Badge>
                              </div>
                              <div className="space-y-1 text-sm text-muted-foreground">
                                <div className="flex items-center gap-2">
                                  <Calendar className="h-4 w-4" />
                                  <span>
                                    {new Date(
                                      appointment.date,
                                    ).toLocaleDateString("en-US", {
                                      weekday: "long",
                                      year: "numeric",
                                      month: "long",
                                      day: "numeric",
                                    })}
                                  </span>
                                </div>
                                <div className="flex items-center gap-2">
                                  <Clock className="h-4 w-4" />
                                  <span>{appointment.time}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                  <User className="h-4 w-4" />
                                  <span>Dr. {appointment.doctorName}</span>
                                </div>
                                {appointment.location && (
                                  <div className="flex items-center gap-2">
                                    <MapPin className="h-4 w-4" />
                                    <span>{appointment.location}</span>
                                  </div>
                                )}
                              </div>
                            </div>
                            {appointment.status === "scheduled" &&
                              new Date(appointment.date) > new Date() && (
                                <div className="flex flex-col gap-2">
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => {
                                      setShowAppointmentsDialog(false);
                                      navigate("/schedule-appointment");
                                    }}
                                    className="text-xs"
                                  >
                                    Reschedule
                                  </Button>
                                </div>
                              )}
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <Calendar className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-healthcare-text mb-2">
                  No Appointments Yet
                </h3>
                <p className="text-muted-foreground mb-4">
                  You haven't scheduled any appointments. Get started by booking
                  your first consultation.
                </p>
                <Button
                  onClick={() => {
                    setShowAppointmentsDialog(false);
                    navigate("/schedule-appointment");
                  }}
                  className="bg-medical-blue hover:bg-medical-blue/90"
                >
                  <Plus className="mr-2 h-4 w-4" />
                  Book Your First Appointment
                </Button>
              </div>
            )}

            <div className="flex justify-end">
              <Button
                variant="outline"
                onClick={() => setShowAppointmentsDialog(false)}
              >
                Close
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* PDF Download Dialog */}
      <Dialog open={showPDFDialog} onOpenChange={setShowPDFDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader className="text-center">
            <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Download className="h-8 w-8 text-blue-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              Download Receipt
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Generate a PDF receipt of your healthcare profile and information.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-medium text-healthcare-text mb-2">
                Your receipt will include:
              </h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Personal profile information</li>
                <li>• Healthcare preferences</li>
                <li>• Appointment details</li>
                <li>• Contact information</li>
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
                onClick={handleDownloadReceipt}
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
                    Generate
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
        <DialogContent className="sm:max-w-lg">
          <DialogHeader className="text-center">
            <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Phone className="h-8 w-8 text-green-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              Schedule Confirmation Call
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Choose your preferred time for our team to call you.
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
            </div>

            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={() => setShowCallScheduleDialog(false)}
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
              <Bell className="h-8 w-8 text-blue-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              Notification Preferences
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Set your preferences for receiving health updates.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6">
            <div className="space-y-3">
              <label className="text-sm font-medium text-healthcare-text">
                Notification Types
              </label>
              <div className="space-y-2">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
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
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
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
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
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
                  { value: "email", label: "Email", icon: Mail },
                  { value: "sms", label: "SMS", icon: Phone },
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
                      onClick={() => {
                        if (method.value === "both") {
                          setLocalDeliveryMethods(["email", "sms"]);
                        } else if (method.value === "email") {
                          setLocalDeliveryMethods(["email"]);
                        } else {
                          setLocalDeliveryMethods(["sms"]);
                        }
                      }}
                    >
                      <method.icon className="h-4 w-4" />
                      <span className="text-xs">{method.label}</span>
                    </Button>
                  );
                })}
              </div>
            </div>

            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={() => setShowNotificationDialog(false)}
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
                    <Bell className="h-4 w-4" />
                    Save
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
              Wellness Program
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Join our wellness program for personalized health tracking.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6">
            <div className="space-y-3">
              <label className="text-sm font-medium text-healthcare-text">
                Choose Your Plan
              </label>
              <div className="grid md:grid-cols-2 gap-4">
                <div
                  className={`relative p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                    localWellnessPlan === "basic"
                      ? "border-green-500 bg-green-50 shadow-md"
                      : "border-gray-200 bg-white hover:border-green-300"
                  }`}
                  onClick={() => setLocalWellnessPlan("basic")}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-healthcare-text">
                      Basic Plan
                    </h3>
                    <Badge className="bg-green-100 text-green-800 font-bold">
                      FREE
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Perfect for getting started with health tracking
                  </p>
                </div>

                <div
                  className={`relative p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                    localWellnessPlan === "premium"
                      ? "border-blue-500 bg-blue-50 shadow-md"
                      : "border-gray-200 bg-white hover:border-blue-300"
                  }`}
                  onClick={() => setLocalWellnessPlan("premium")}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-healthcare-text">
                      Premium Plan
                    </h3>
                    <Badge className="bg-blue-100 text-blue-800 font-bold">
                      $29/mo
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Complete wellness support with personal coaching
                  </p>
                </div>
              </div>
            </div>

            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={() => setShowWellnessDialog(false)}
                className="flex-1"
                disabled={isEnrollingWellness}
              >
                Later
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
                    {localWellnessPlan === "premium" ? "Start Trial" : "Enroll"}
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
