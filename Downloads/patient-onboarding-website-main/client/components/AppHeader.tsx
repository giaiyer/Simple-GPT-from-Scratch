import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useState } from "react";
import { useUser } from "@/lib/UserContext";
import {
  Heart,
  ArrowLeft,
  LayoutDashboard,
  Calendar,
  Clock,
  MapPin,
  User,
  Phone,
} from "lucide-react";

interface AppHeaderProps {
  showBackButton?: boolean;
  backButtonText?: string;
  backButtonAction?: () => void;
  adminMode?: boolean;
}

export default function AppHeader({
  showBackButton = false,
  backButtonText = "Back",
  backButtonAction,
  adminMode = false,
}: AppHeaderProps) {
  const navigate = useNavigate();
  const { userData, appointments, logoutUser } = useUser();
  const [showAppointments, setShowAppointments] = useState(false);

  const firstName = userData?.firstName || "Patient";

  const handleBackClick = () => {
    if (backButtonAction) {
      backButtonAction();
    } else {
      navigate(-1);
    }
  };

  const handleSignOut = () => {
    logoutUser();
    navigate("/");
  };

  const upcomingAppointments = appointments.filter(
    (apt) => apt.status === "scheduled" && new Date(apt.date) >= new Date(),
  );

  return (
    <>
      <header className="bg-white/80 backdrop-blur-sm border-b border-medical-blue/20 sticky top-0 z-40">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-medical-blue p-2 rounded-lg">
              <Heart className="h-6 w-6 text-white" />
            </div>
            <span className="text-xl font-bold text-healthcare-text">
              HealthCare Plus
            </span>
          </div>

          <div className="flex items-center gap-3">
            {adminMode ? (
              // Admin mode: only show Sign Out
              <Button
                variant="ghost"
                onClick={handleSignOut}
                className="text-healthcare-text hover:text-red-600"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Sign Out
              </Button>
            ) : (
              <>
                {userData && (
                  <>
                    <Button
                      variant="ghost"
                      onClick={() => navigate("/dashboard")}
                      className="text-healthcare-text hover:text-medical-blue hover:bg-medical-light-blue/20"
                    >
                      <LayoutDashboard className="mr-2 h-4 w-4" />
                      Dashboard
                    </Button>

                    <Button
                      variant="ghost"
                      onClick={() => setShowAppointments(true)}
                      className="text-healthcare-text hover:text-medical-green hover:bg-medical-light-green/20 relative"
                    >
                      <Calendar className="mr-2 h-4 w-4" />
                      My Appointments
                      {upcomingAppointments.length > 0 && (
                        <span className="absolute -top-1 -right-1 bg-medical-green text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                          {upcomingAppointments.length}
                        </span>
                      )}
                    </Button>
                  </>
                )}

                {showBackButton && (
                  <Button
                    variant="ghost"
                    onClick={handleBackClick}
                    className="text-healthcare-text hover:text-medical-blue"
                  >
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    {backButtonText}
                  </Button>
                )}

                {userData ? (
                  <Button
                    variant="ghost"
                    onClick={handleSignOut}
                    className="text-healthcare-text hover:text-red-600"
                  >
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Sign Out
                  </Button>
                ) : (
                  <Button
                    onClick={() => navigate("/login")}
                    className="bg-medical-blue hover:bg-medical-blue/90 text-white"
                  >
                    Sign In
                  </Button>
                )}
              </>
            )}
          </div>
        </div>
      </header>

      {/* My Appointments Modal */}
      <Dialog open={showAppointments} onOpenChange={setShowAppointments}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-xl font-bold text-healthcare-text flex items-center gap-2">
              <Calendar className="h-5 w-5 text-medical-green" />
              My Appointments
            </DialogTitle>
            <DialogDescription>
              View your scheduled and past appointments
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 max-h-96 overflow-y-auto">
            {appointments.length === 0 ? (
              <div className="text-center py-8">
                <Calendar className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-healthcare-text mb-2">
                  No Appointments Yet
                </h3>
                <p className="text-muted-foreground mb-4">
                  You haven't scheduled any appointments yet.
                </p>
                <Button
                  onClick={() => {
                    setShowAppointments(false);
                    navigate("/schedule-appointment");
                  }}
                  className="bg-medical-blue hover:bg-medical-blue/90"
                >
                  <Calendar className="mr-2 h-4 w-4" />
                  Schedule Appointment
                </Button>
              </div>
            ) : (
              <>
                {appointments.map((appointment) => (
                  <div
                    key={appointment.id}
                    className={`border rounded-lg p-4 ${
                      appointment.status === "scheduled"
                        ? "border-green-200 bg-green-50"
                        : appointment.status === "completed"
                          ? "border-blue-200 bg-blue-50"
                          : "border-red-200 bg-red-50"
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <h4 className="font-semibold text-healthcare-text">
                            {appointment.appointmentType}
                          </h4>
                          <span
                            className={`px-2 py-1 rounded-full text-xs font-medium ${
                              appointment.status === "scheduled"
                                ? "bg-green-100 text-green-800"
                                : appointment.status === "completed"
                                  ? "bg-blue-100 text-blue-800"
                                  : "bg-red-100 text-red-800"
                            }`}
                          >
                            {appointment.status.charAt(0).toUpperCase() +
                              appointment.status.slice(1)}
                          </span>
                        </div>

                        <div className="space-y-1 text-sm text-muted-foreground">
                          <div className="flex items-center gap-2">
                            <Calendar className="h-4 w-4" />
                            <span>
                              {new Date(appointment.date).toLocaleDateString(
                                "en-US",
                                {
                                  weekday: "long",
                                  year: "numeric",
                                  month: "long",
                                  day: "numeric",
                                },
                              )}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Clock className="h-4 w-4" />
                            <span>{appointment.time}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <User className="h-4 w-4" />
                            <span>{appointment.doctorName}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <MapPin className="h-4 w-4" />
                            <span>{appointment.location}</span>
                          </div>
                        </div>
                      </div>

                      {appointment.status === "scheduled" && (
                        <div className="flex flex-col gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              setShowAppointments(false);
                              navigate(
                                `/schedule-appointment?reschedule=${appointment.id}`,
                              );
                            }}
                            className="text-xs"
                          >
                            Reschedule
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                <div className="pt-4 border-t">
                  <Button
                    onClick={() => {
                      setShowAppointments(false);
                      navigate("/schedule-appointment");
                    }}
                    className="w-full bg-medical-blue hover:bg-medical-blue/90"
                  >
                    <Calendar className="mr-2 h-4 w-4" />
                    Schedule New Appointment
                  </Button>
                </div>
              </>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
