import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { UserProvider } from "@/lib/UserContext";
import Index from "./pages/Index";
import Login from "./pages/Login";
import Register from "./pages/Register";
import ProfileSetup from "./pages/ProfileSetup";
import Dashboard from "./pages/Dashboard";
import UploadDocuments from "./pages/UploadDocuments";
import ScheduleAppointment from "./pages/ScheduleAppointment";
import ConsentForms from "./pages/ConsentForms";
import MedicalHistory from "./pages/MedicalHistory";
import EmergencyContacts from "./pages/EmergencyContacts";
import OnboardingComplete from "./pages/OnboardingComplete";
import ProfileEdit from "./pages/ProfileEdit";
import EHRDashboard from "./pages/EHRDashboard";
import AdminLogin from "./pages/AdminLogin";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <UserProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/profile-setup" element={<ProfileSetup />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/upload-documents" element={<UploadDocuments />} />
            <Route
              path="/schedule-appointment"
              element={<ScheduleAppointment />}
            />
            <Route path="/consent-forms" element={<ConsentForms />} />
            <Route path="/medical-history" element={<MedicalHistory />} />
            <Route path="/emergency-contacts" element={<EmergencyContacts />} />
            <Route
              path="/onboarding-complete"
              element={<OnboardingComplete />}
            />
            <Route path="/profile-edit" element={<ProfileEdit />} />
            <Route path="/ehr-dashboard" element={<EHRDashboard />} />
            <Route path="/admin-login" element={<AdminLogin />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </UserProvider>
  </QueryClientProvider>
);

createRoot(document.getElementById("root")!).render(<App />);
