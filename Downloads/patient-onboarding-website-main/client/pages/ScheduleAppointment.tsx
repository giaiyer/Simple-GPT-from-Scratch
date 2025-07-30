import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useUser, AppointmentData } from "@/lib/UserContext";
import AppHeader from "@/components/AppHeader";
import { Button } from "@/components/ui/button";
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
import { Calendar } from "@/components/ui/calendar";
import {
  Heart,
  ArrowLeft,
  Calendar as CalendarIcon,
  Clock,
  User,
  Mail,
  CheckCircle2,
  Download,
  Star,
  MapPin,
  Phone,
  ArrowRight,
} from "lucide-react";

interface TimeSlot {
  time: string;
  available: boolean;
  doctor?: string;
}

interface AppointmentType {
  id: string;
  name: string;
  duration: number;
  description: string;
  //price: string;
}

interface Doctor {
  id: string;
  name: string;
  specialty: string;
  rating: number;
  image: string;
}

interface AppointmentData {
  date: Date | undefined;
  appointmentType: string;
  timeSlot: string;
  doctor: string;
}

export default function ScheduleAppointment() {
  const navigate = useNavigate();
  const {
    updateOnboardingProgress,
    addAppointment,
    updateAppointment,
    getCompletionPercentage,
  } = useUser();

  // Check if we're in reschedule mode
  const urlParams = new URLSearchParams(window.location.search);
  const rescheduleId = urlParams.get("reschedule");
  const isRescheduling = !!rescheduleId;

  // Check if onboarding is complete
  const isOnboardingComplete = getCompletionPercentage() === 100;
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(undefined);
  const [selectedAppointmentType, setSelectedAppointmentType] = useState("");
  const [selectedTimeSlot, setSelectedTimeSlot] = useState("");
  const [selectedDoctor, setSelectedDoctor] = useState("");
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [showEmailSent, setShowEmailSent] = useState(false);
  const [appointmentData, setAppointmentData] =
    useState<AppointmentData | null>(null);

  const appointmentTypes: AppointmentType[] = [
    {
      id: "annual-checkup",
      name: "Annual Checkup",
      duration: 60,
      description: "Comprehensive health assessment and preventive care",
      //price: "$250",
    },
    {
      id: "consultation",
      name: "General Consultation",
      duration: 30,
      description: "Discussion of health concerns with a healthcare provider",
      //price: "$150",
    },
    {
      id: "follow-up",
      name: "Follow-up Visit",
      duration: 30,
      description: "Review of previous treatment or test results",
      //price: "$120",
    },
    {
      id: "urgent-care",
      name: "Urgent Care",
      duration: 45,
      description: "Same-day care for non-emergency medical needs",
      //price: "$200",
    },
  ];

  const doctors: Doctor[] = [
    {
      id: "dr-smith",
      name: "Dr. Sarah Smith",
      specialty: "Family Medicine",
      rating: 4.9,
      image: "/api/placeholder/64/64",
    },
    {
      id: "dr-johnson",
      name: "Dr. Michael Johnson",
      specialty: "Internal Medicine",
      rating: 4.8,
      image: "/api/placeholder/64/64",
    },
    {
      id: "dr-williams",
      name: "Dr. Emily Williams",
      specialty: "Preventive Care",
      rating: 4.9,
      image: "/api/placeholder/64/64",
    },
  ];

  // Generate time slots for a day and specific doctor
  const generateTimeSlots = (date: Date, doctorId?: string): TimeSlot[] => {
    const slots: TimeSlot[] = [];
    const today = new Date();
    const isToday =
      date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear();

    const startHour = 9; // 9 AM
    const endHour = 17; // 5 PM
    const selectedDoctorData = doctorId
      ? doctors.find((d) => d.id === doctorId)
      : null;

    for (let hour = startHour; hour < endHour; hour++) {
      for (let minute = 0; minute < 60; minute += 30) {
        const timeSlot = `${hour.toString().padStart(2, "0")}:${minute.toString().padStart(2, "0")}`;

        // If it's today, disable past time slots
        let available = true;
        if (isToday) {
          const slotTime = new Date(date);
          slotTime.setHours(hour, minute, 0, 0);
          available = slotTime > today;
        }

        // If a doctor is selected, filter slots for that doctor only
        if (available && doctorId && selectedDoctorData) {
          // Create deterministic availability based on doctor and time slot
          const slotHash = `${doctorId}-${date.toDateString()}-${timeSlot}`;
          const hash = slotHash
            .split("")
            .reduce((a, b) => ((a << 5) - a + b.charCodeAt(0)) | 0, 0);
          available = Math.abs(hash) % 100 < 70; // 70% availability for selected doctor
        } else if (available && !doctorId) {
          // If no doctor selected, randomly make some slots unavailable
          available = Math.random() < 0.7;
        }

        slots.push({
          time: timeSlot,
          available,
          doctor:
            available && selectedDoctorData
              ? selectedDoctorData.name
              : undefined,
        });
      }
    }

    return slots;
  };

  const [timeSlots, setTimeSlots] = useState<TimeSlot[]>([]);

  useEffect(() => {
    if (selectedDate) {
      setTimeSlots(generateTimeSlots(selectedDate, selectedDoctor));
      setSelectedTimeSlot(""); // Reset time slot when date changes
    }
  }, [selectedDate, selectedDoctor]);

  const isDateUnavailable = (date: Date) => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Disable past dates
    if (date < today) return true;

    // Disable weekends (just for demo)
    const dayOfWeek = date.getDay();
    if (dayOfWeek === 0 || dayOfWeek === 6) return true;

    // Randomly disable some dates (simulate unavailable dates)
    const dateString = date.toDateString();
    const hash = dateString
      .split("")
      .reduce((a, b) => ((a << 5) - a + b.charCodeAt(0)) | 0, 0);
    return Math.abs(hash) % 4 === 0;
  };

  const canProceed = () => {
    return (
      selectedDate &&
      selectedAppointmentType &&
      selectedTimeSlot &&
      selectedDoctor
    );
  };

  const handleSchedule = () => {
    if (!canProceed()) return;

    const appointmentTypeData = appointmentTypes.find(
      (type) => type.id === selectedAppointmentType,
    );
    const doctorData = doctors.find((doc) => doc.id === selectedDoctor);

    const data: AppointmentData = {
      date: selectedDate,
      appointmentType: selectedAppointmentType,
      timeSlot: selectedTimeSlot,
      doctor: selectedDoctor,
    };

    setAppointmentData(data);
    setShowConfirmation(true);
  };

  const handleConfirmAppointment = async () => {
    if (!appointmentData || !selectedDate) return;

    const appointmentTypeData = appointmentTypes.find(
      (type) => type.id === appointmentData.appointmentType,
    );
    const doctorData = doctors.find((doc) => doc.id === appointmentData.doctor);

    try {
      const token = localStorage.getItem("healthcarePlus_token");

      if (isRescheduling && rescheduleId) {
        // Update existing appointment
        const updatedAppointment = {
          date: selectedDate.toISOString().split("T")[0],
          time: appointmentData.timeSlot,
          doctorName: doctorData?.name || "Unknown Doctor",
          appointmentType: appointmentTypeData?.name || "Unknown Type",
          status: "scheduled",
          location: "HealthCare Plus Main Clinic",
        };

        const response = await fetch(`/api/medical/appointments/${rescheduleId}`, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`,
          },
          body: JSON.stringify(updatedAppointment),
        });

        if (response.ok) {
          updateAppointment(rescheduleId, updatedAppointment);
        } else {
          throw new Error("Failed to update appointment");
        }
      } else {
        // Create new appointment
        const newAppointment = {
          date: selectedDate.toISOString().split("T")[0],
          time: appointmentData.timeSlot,
          doctorName: doctorData?.name || "Unknown Doctor",
          appointmentType: appointmentTypeData?.name || "Unknown Type",
          location: "HealthCare Plus Main Clinic",
        };

        const response = await fetch("/api/medical/appointments", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`,
          },
          body: JSON.stringify(newAppointment),
        });

        if (response.ok) {
          const data = await response.json();
          addAppointment({
            id: data.appointment.id.toString(),
            date: data.appointment.date,
            time: data.appointment.time,
            doctorName: data.appointment.doctorName,
            appointmentType: data.appointment.appointmentType,
            status: data.appointment.status,
            location: data.appointment.location,
          });
          updateOnboardingProgress("scheduleAppointment", true);
        } else {
          throw new Error("Failed to create appointment");
        }
      }

      setShowConfirmation(false);
      setShowEmailSent(true);

      // Auto-hide email notification after 3 seconds
      setTimeout(() => {
        setShowEmailSent(false);
      }, 3000);
    } catch (error) {
      console.error("Error with appointment:", error);
      alert("Failed to save appointment. Please try again.");
    }
  };

  const addToGoogleCalendar = () => {
    if (!appointmentData || !selectedDate) return;

    const appointmentTypeData = appointmentTypes.find(
      (type) => type.id === appointmentData.appointmentType,
    );
    const doctorData = doctors.find((doc) => doc.id === appointmentData.doctor);

    const startDate = new Date(selectedDate);
    const [hours, minutes] = appointmentData.timeSlot.split(":").map(Number);
    startDate.setHours(hours, minutes, 0, 0);

    const endDate = new Date(startDate);
    endDate.setMinutes(
      endDate.getMinutes() + (appointmentTypeData?.duration || 30),
    );

    const title = encodeURIComponent(
      `${appointmentTypeData?.name} with ${doctorData?.name}`,
    );
    const details = encodeURIComponent(
      `Appointment Type: ${appointmentTypeData?.name}\nDoctor: ${doctorData?.name}\nSpecialty: ${doctorData?.specialty}\nLocation: HealthCare Plus Main Clinic`,
    );
    const location = encodeURIComponent("HealthCare Plus Main Clinic");

    const googleCalendarUrl = `https://calendar.google.com/calendar/render?action=TEMPLATE&text=${title}&dates=${startDate
      .toISOString()
      .replace(/[-:]/g, "")
      .replace(/\.\d{3}/, "")}/${endDate
      .toISOString()
      .replace(/[-:]/g, "")
      .replace(/\.\d{3}/, "")}&details=${details}&location=${location}`;

    window.open(googleCalendarUrl, "_blank");
  };

  const addToOutlookCalendar = () => {
    if (!appointmentData || !selectedDate) return;

    const appointmentTypeData = appointmentTypes.find(
      (type) => type.id === appointmentData.appointmentType,
    );
    const doctorData = doctors.find((doc) => doc.id === appointmentData.doctor);

    const startDate = new Date(selectedDate);
    const [hours, minutes] = appointmentData.timeSlot.split(":").map(Number);
    startDate.setHours(hours, minutes, 0, 0);

    const endDate = new Date(startDate);
    endDate.setMinutes(
      endDate.getMinutes() + (appointmentTypeData?.duration || 30),
    );

    const title = encodeURIComponent(
      `${appointmentTypeData?.name} with ${doctorData?.name}`,
    );
    const body = encodeURIComponent(
      `Appointment Type: ${appointmentTypeData?.name}\nDoctor: ${doctorData?.name}\nSpecialty: ${doctorData?.specialty}\nLocation: HealthCare Plus Main Clinic`,
    );
    const location = encodeURIComponent("HealthCare Plus Main Clinic");

    const outlookUrl = `https://outlook.live.com/calendar/0/deeplink/compose?subject=${title}&startdt=${startDate.toISOString()}&enddt=${endDate.toISOString()}&body=${body}&location=${location}`;

    window.open(outlookUrl, "_blank");
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const formatTime = (time: string) => {
    const [hours, minutes] = time.split(":").map(Number);
    const date = new Date();
    date.setHours(hours, minutes);
    return date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light-blue via-background to-medical-light-green">
      <AppHeader
        showBackButton={true}
        backButtonText={
          isOnboardingComplete ? "Back to Dashboard" : "Back to Onboarding"
        }
        backButtonAction={() =>
          navigate(isOnboardingComplete ? "/dashboard" : "/profile-setup")
        }
      />

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {/* Page Header */}
          <div className="text-center mb-8">
            <div className="bg-medical-green/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <CalendarIcon className="h-8 w-8 text-medical-green" />
            </div>
            <h1 className="text-3xl font-bold text-healthcare-text mb-2">
              {isRescheduling
                ? "Reschedule Your Appointment"
                : "Schedule Your Appointment"}
            </h1>
            <p className="text-lg text-muted-foreground">
              {isRescheduling
                ? "Select a new date and time for your appointment"
                : "Book a consultation with one of our healthcare providers"}
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Left Column - Selection */}
            <div className="space-y-6">
              {/* Appointment Type */}
              <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-lg text-healthcare-text flex items-center gap-2">
                    <User className="h-5 w-5" />
                    Select Appointment Type
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {appointmentTypes.map((type) => (
                      <div
                        key={type.id}
                        className={`border rounded-lg p-4 cursor-pointer transition-all duration-200 ${
                          selectedAppointmentType === type.id
                            ? "border-medical-blue bg-medical-light-blue/20"
                            : "border-gray-200 hover:border-medical-blue/50"
                        }`}
                        onClick={() => setSelectedAppointmentType(type.id)}
                      >
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <h3 className="font-semibold text-healthcare-text">
                              {type.name}
                            </h3>
                            <p className="text-sm text-muted-foreground mt-1">
                              {type.description}
                            </p>
                            <div className="flex items-center gap-2 mt-2">
                              <Badge variant="outline">
                                <Clock className="h-3 w-3 mr-1" />
                                {type.duration} min
                              </Badge>
                            </div>
                          </div>
                          <div className="text-right">
                            <span className="font-bold text-medical-green">
                              {type.price}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Doctor Selection */}
              {selectedAppointmentType && (
                <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-lg text-healthcare-text flex items-center gap-2">
                      <User className="h-5 w-5" />
                      Choose Your Doctor
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {doctors.map((doctor) => (
                        <div
                          key={doctor.id}
                          className={`border rounded-lg p-4 cursor-pointer transition-all duration-200 ${
                            selectedDoctor === doctor.id
                              ? "border-medical-green bg-medical-light-green/20"
                              : "border-gray-200 hover:border-medical-green/50"
                          }`}
                          onClick={() => setSelectedDoctor(doctor.id)}
                        >
                          <div className="flex items-center gap-3">
                            <div className="w-12 h-12 bg-medical-blue/10 rounded-full flex items-center justify-center">
                              <User className="h-6 w-6 text-medical-blue" />
                            </div>
                            <div className="flex-1">
                              <h3 className="font-semibold text-healthcare-text">
                                {doctor.name}
                              </h3>
                              <p className="text-sm text-muted-foreground">
                                {doctor.specialty}
                              </p>
                              <div className="flex items-center gap-1 mt-1">
                                <Star className="h-4 w-4 text-yellow-500 fill-current" />
                                <span className="text-sm font-medium">
                                  {doctor.rating}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Right Column - Calendar and Time Slots */}
            <div className="space-y-6">
              {/* Calendar */}
              <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-lg text-healthcare-text flex items-center gap-2">
                    <CalendarIcon className="h-5 w-5" />
                    Select Date
                  </CardTitle>
                  <CardDescription>
                    Choose an available date for your appointment
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Calendar
                    mode="single"
                    selected={selectedDate}
                    onSelect={setSelectedDate}
                    disabled={isDateUnavailable}
                    className="rounded-md border"
                  />
                  <div className="mt-4 text-xs text-muted-foreground">
                    <p>• Weekends are not available</p>
                    <p>• Some dates may be fully booked</p>
                  </div>
                </CardContent>
              </Card>

              {/* Time Slots */}
              {selectedDate && selectedDoctor && (
                <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-lg text-healthcare-text flex items-center gap-2">
                      <Clock className="h-5 w-5" />
                      Available Time Slots
                    </CardTitle>
                    <CardDescription>
                      {formatDate(selectedDate)} -{" "}
                      {doctors.find((d) => d.id === selectedDoctor)?.name}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-2 max-h-60 overflow-y-auto">
                      {timeSlots.map((slot) => (
                        <Button
                          key={slot.time}
                          variant={
                            selectedTimeSlot === slot.time
                              ? "default"
                              : "outline"
                          }
                          disabled={!slot.available}
                          onClick={() => setSelectedTimeSlot(slot.time)}
                          className={`h-auto py-3 ${
                            selectedTimeSlot === slot.time
                              ? "bg-medical-green hover:bg-medical-green/90"
                              : "hover:bg-medical-light-blue/20"
                          }`}
                        >
                          <div className="text-center">
                            <div className="font-semibold">
                              {formatTime(slot.time)}
                            </div>
                          </div>
                        </Button>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>

          {/* Schedule Button */}
          {canProceed() && (
            <div className="mt-8 text-center">
              <Button
                onClick={handleSchedule}
                className="bg-medical-blue hover:bg-medical-blue/90 text-white px-8 py-3 text-lg font-semibold"
              >
                <CalendarIcon className="mr-2 h-5 w-5" />
                Schedule Appointment
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Confirmation Modal */}
      <Dialog open={showConfirmation} onOpenChange={setShowConfirmation}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader className="text-center">
            <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle2 className="h-8 w-8 text-green-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              {isRescheduling
                ? "Confirm Rescheduled Appointment"
                : "Confirm Your Appointment"}
            </DialogTitle>
          </DialogHeader>

          {appointmentData && selectedDate && (
            <div className="space-y-4">
              <div className="bg-medical-light-blue/20 p-4 rounded-lg space-y-3">
                <div className="flex items-center gap-2">
                  <CalendarIcon className="h-4 w-4 text-medical-blue" />
                  <span className="font-medium">
                    {formatDate(selectedDate)}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-medical-blue" />
                  <span className="font-medium">
                    {formatTime(appointmentData.timeSlot)}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <User className="h-4 w-4 text-medical-blue" />
                  <span className="font-medium">
                    {doctors.find((d) => d.id === appointmentData.doctor)?.name}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-medical-blue" />
                  <span className="font-medium">
                    HealthCare Plus Main Clinic
                  </span>
                </div>
              </div>

              <div className="text-sm text-muted-foreground space-y-1">
                <p>
                  <strong>Appointment Type:</strong>{" "}
                  {
                    appointmentTypes.find(
                      (t) => t.id === appointmentData.appointmentType,
                    )?.name
                  }
                </p>
                <p>
                  <strong>Duration:</strong>{" "}
                  {
                    appointmentTypes.find(
                      (t) => t.id === appointmentData.appointmentType,
                    )?.duration
                  }{" "}
                  minutes
                </p>
                <p></p>
              </div>

              <div className="flex gap-3">
                <Button
                  variant="outline"
                  onClick={() => setShowConfirmation(false)}
                  className="flex-1"
                >
                  Back
                </Button>
                <Button
                  onClick={handleConfirmAppointment}
                  className="flex-1 bg-medical-green hover:bg-medical-green/90"
                >
                  Confirm Appointment
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Email Sent Notification */}
      <Dialog open={showEmailSent} onOpenChange={setShowEmailSent}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader className="text-center">
            <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Mail className="h-8 w-8 text-blue-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text">
              {isRescheduling ? "Appointment Rescheduled!" : "Email Sent!"}
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              {isRescheduling
                ? "Your appointment has been successfully rescheduled. A confirmation email has been sent with the updated details."
                : "Confirmation email has been sent to your registered email address with appointment details and reminders."}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="flex flex-col gap-2">
              <Button
                onClick={addToGoogleCalendar}
                variant="outline"
                className="w-full flex items-center gap-2 border-blue-500 text-blue-600 hover:bg-blue-50"
              >
                <CalendarIcon className="h-4 w-4" />
                Add to Google Calendar
              </Button>
              <Button
                onClick={addToOutlookCalendar}
                variant="outline"
                className="w-full flex items-center gap-2 border-blue-600 text-blue-700 hover:bg-blue-50"
              >
                <CalendarIcon className="h-4 w-4" />
                Add to Outlook Calendar
              </Button>
            </div>

            <div className="flex gap-3">
              <Button
                onClick={() =>
                  navigate(
                    isRescheduling || isOnboardingComplete
                      ? "/dashboard"
                      : "/profile-setup",
                  )
                }
                className="flex-1 bg-medical-blue hover:bg-medical-blue/90"
              >
                {isRescheduling || isOnboardingComplete
                  ? "Back to Dashboard"
                  : "Continue Onboarding"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
