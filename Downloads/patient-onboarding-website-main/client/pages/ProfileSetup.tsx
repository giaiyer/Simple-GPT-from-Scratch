import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "@/lib/UserContext";
import AppHeader from "@/components/AppHeader";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Heart,
  ArrowLeft,
  CheckCircle2,
  Clock,
  X,
  Upload,
  Calendar,
  FileText,
  User,
  CreditCard,
  Stethoscope,
  Star,
  Sparkles,
  ArrowRight,
  Home,
} from "lucide-react";

interface ChecklistItem {
  id: string;
  title: string;
  description: string;
  status: "complete" | "pending" | "not-started";
  icon: React.ComponentType<{ className?: string }>;
  route?: string;
}

export default function ProfileSetup() {
  const navigate = useNavigate();
  const {
    userData,
    onboardingProgress,
    updateOnboardingProgress,
    getCompletionPercentage,
    getCompletedTasksCount,
  } = useUser();

  const userName = userData
    ? `${userData.firstName} ${userData.lastName}`
    : "Patient";
  const checklistItems: ChecklistItem[] = [
    {
      id: "upload-documents",
      title: "Upload ID and Insurance Card",
      description: "Verify your identity and insurance coverage",
      status: onboardingProgress.uploadDocuments ? "complete" : "not-started",
      icon: Upload,
      route: "/upload-documents",
    },
    {
      id: "schedule-appointment",
      title: "Schedule Your First Appointment",
      description: "Book a consultation with one of our healthcare providers",
      status: onboardingProgress.scheduleAppointment
        ? "complete"
        : "not-started",
      icon: Calendar,
      route: "/schedule-appointment",
    },
    {
      id: "medical-history",
      title: "Complete Medical History",
      description: "Provide detailed information about your health background",
      status: onboardingProgress.medicalHistory ? "complete" : "not-started",
      icon: Stethoscope,
      route: "/medical-history",
    },
    {
      id: "emergency-contacts",
      title: "Add Emergency Contacts",
      description: "Provide contact information for emergencies",
      status: onboardingProgress.emergencyContacts ? "complete" : "not-started",
      icon: User,
      route: "/emergency-contacts",
    },
    {
      id: "consent-forms",
      title: "Sign Consent Forms",
      description: "Review and sign necessary medical consent documents",
      status: onboardingProgress.consentForms ? "complete" : "not-started",
      icon: FileText,
      route: "/consent-forms",
    },
  ];

  const [celebratingTask, setCelebratingTask] = useState<string | null>(null);
  const [showConfetti, setShowConfetti] = useState(false);

  const completedTasks = getCompletedTasksCount();
  const totalTasks = checklistItems.length;
  const completionPercentage = getCompletionPercentage();

  const handleTaskClick = (item: ChecklistItem) => {
    if (item.route) {
      navigate(item.route);
    }
  };

  const getStatusIcon = (status: ChecklistItem["status"]) => {
    switch (status) {
      case "complete":
        return <CheckCircle2 className="h-5 w-5 text-green-600" />;
      case "pending":
        return <Clock className="h-5 w-5 text-yellow-600" />;
      case "not-started":
        return <X className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusBadge = (status: ChecklistItem["status"]) => {
    switch (status) {
      case "complete":
        return (
          <Badge className="bg-green-100 text-green-800 hover:bg-green-100">
            Complete
          </Badge>
        );
      case "pending":
        return (
          <Badge className="bg-yellow-100 text-yellow-800 hover:bg-yellow-100">
            Pending
          </Badge>
        );
      case "not-started":
        return (
          <Badge className="bg-gray-100 text-gray-600 hover:bg-gray-100">
            Not Started
          </Badge>
        );
    }
  };

  const getProgressColor = () => {
    if (completionPercentage >= 80) return "bg-green-500";
    if (completionPercentage >= 50) return "bg-blue-500";
    if (completionPercentage >= 25) return "bg-yellow-500";
    return "bg-gray-400";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light-blue via-background to-medical-light-green relative overflow-hidden">
      {/* Confetti Animation */}
      {showConfetti && (
        <div className="fixed inset-0 pointer-events-none z-50">
          {Array.from({ length: 50 }).map((_, i) => (
            <div
              key={i}
              className="absolute animate-bounce"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 2}s`,
                animationDuration: `${2 + Math.random() * 2}s`,
              }}
            >
              <Sparkles className="h-4 w-4 text-yellow-400" />
            </div>
          ))}
        </div>
      )}

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
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              onClick={() => navigate("/dashboard")}
              className="text-healthcare-text hover:text-medical-blue"
            >
              <Home className="mr-2 h-4 w-4" />
              Dashboard
            </Button>
            <Button
              variant="ghost"
              onClick={() => navigate("/")}
              className="text-healthcare-text hover:text-medical-blue"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Sign Out
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        {/* Welcome Section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 bg-medical-light-green/50 px-4 py-2 rounded-full text-medical-green font-medium mb-4">
            <Star className="h-4 w-4" />
            Welcome to HealthCare Plus!
          </div>
          <h1 className="text-4xl font-bold text-healthcare-text mb-2">
            Hi {userName.split(" ")[0]}! Let's get you started.
          </h1>
          <p className="text-lg text-muted-foreground">
            Complete these quick steps to set up your healthcare profile
          </p>
        </div>

        {/* Progress Overview */}
        <div className="max-w-4xl mx-auto mb-8">
          <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
            <CardHeader className="text-center">
              <CardTitle className="text-2xl text-healthcare-text mb-2">
                Onboarding Progress
              </CardTitle>
              <CardDescription className="text-base">
                You're {completionPercentage}% complete!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-healthcare-text font-medium">
                    Overall Progress
                  </span>
                  <span className="text-healthcare-text font-bold">
                    {completedTasks} of {totalTasks} tasks completed
                  </span>
                </div>
                <div className="relative">
                  <Progress value={completionPercentage} className="h-3" />
                  <div
                    className={`absolute top-0 left-0 h-3 rounded-full transition-all duration-500 ${getProgressColor()}`}
                    style={{ width: `${completionPercentage}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-center">
                <div className="bg-medical-light-green/20 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-medical-green">
                    {completedTasks}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Tasks Complete
                  </div>
                </div>
                <div className="bg-orange-100 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">
                    {totalTasks - completedTasks}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Tasks Remaining
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Checklist */}
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold text-healthcare-text mb-6 text-center">
            Onboarding Checklist
          </h2>

          <div className="space-y-4">
            {checklistItems.map((item, index) => (
              <Card
                key={item.id}
                className={`transition-all duration-300 cursor-pointer hover:shadow-lg border-l-4 ${
                  item.status === "complete"
                    ? "border-l-green-500 bg-green-50/50"
                    : item.status === "pending"
                      ? "border-l-yellow-500 bg-yellow-50/50"
                      : "border-l-gray-300 bg-white/90"
                } ${celebratingTask === item.id ? "animate-pulse ring-4 ring-green-300" : ""} backdrop-blur-sm`}
                onClick={() => handleTaskClick(item)}
              >
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 flex-1">
                      <div className="flex items-center gap-3">
                        <div className="bg-medical-blue/10 p-3 rounded-full">
                          <item.icon className="h-6 w-6 text-medical-blue" />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h3 className="font-semibold text-healthcare-text">
                              {item.title}
                            </h3>
                            {celebratingTask === item.id && (
                              <Sparkles className="h-4 w-4 text-yellow-500 animate-spin" />
                            )}
                          </div>
                          <p className="text-muted-foreground text-sm">
                            {item.description}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-xs text-muted-foreground">
                          Task {index + 1} of {totalTasks}
                        </div>
                      </div>

                      <div className="flex items-center gap-3">
                        {getStatusBadge(item.status)}
                        {getStatusIcon(item.status)}

                        <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Completion Message */}
        {completionPercentage === 100 && (
          <div className="max-w-2xl mx-auto mt-8">
            <Card className="border-0 shadow-xl bg-gradient-to-r from-green-50 to-medical-light-green/50">
              <CardContent className="p-8 text-center">
                <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                  <CheckCircle2 className="h-8 w-8 text-green-600" />
                </div>
                <h3 className="text-2xl font-bold text-healthcare-text mb-2">
                  Congratulations!
                </h3>
                <p className="text-muted-foreground mb-6">
                  You've completed your onboarding! Your healthcare profile is
                  now ready, and you can access all our services.
                </p>
                <Button
                  onClick={() => navigate("/onboarding-complete")}
                  className="bg-medical-green hover:bg-medical-green/90 text-white px-8"
                >
                  <Star className="mr-2 h-4 w-4" />
                  View Completion Summary
                </Button>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
