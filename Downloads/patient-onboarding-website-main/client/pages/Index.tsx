import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Heart,
  UserPlus,
  Calendar,
  Shield,
  Phone,
  Mail,
  MapPin,
  Stethoscope,
  Users,
  CheckCircle2,
  Clock,
} from "lucide-react";

export default function Index() {
  const navigate = useNavigate();

  const handleLogin = () => {
    navigate("/login");
  };

  const handleRegister = () => {
    navigate("/register");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light-blue via-background to-medical-light-green">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-medical-blue/20 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-medical-blue p-2 rounded-lg">
              <Heart className="h-6 w-6 text-white" />
            </div>
            <span className="text-xl font-bold text-healthcare-text">
              HealthCare Plus
            </span>
          </div>
          <div className="flex gap-3">
            <Button
              variant="outline"
              onClick={handleLogin}
              className="border-medical-blue text-medical-blue hover:bg-medical-blue hover:text-white transition-all duration-300"
            >
              Log In
            </Button>
            <Button
              onClick={handleRegister}
              className="bg-medical-blue hover:bg-medical-blue/90 transition-all duration-300 shadow-lg hover:shadow-xl"
            >
              Register Now
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16 text-center">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <div className="inline-flex items-center gap-2 bg-medical-light-green/50 px-4 py-2 rounded-full text-medical-green font-medium mb-6">
              <Stethoscope className="h-4 w-4" />
              Trusted Healthcare Partner
            </div>
          </div>

          <h1 className="text-5xl md:text-6xl font-bold text-healthcare-text mb-6 leading-tight">
            Welcome to{" "}
            <span className="text-medical-blue">HealthCare Plus</span>
            <br />
            <span className="text-3xl md:text-4xl text-medical-green">
              Start Your Healthcare Journey
            </span>
          </h1>

          <p className="text-xl text-muted-foreground mb-12 max-w-2xl mx-auto leading-relaxed">
            Experience truly seamless and personalized healthcare from day one. Your well-being is our priority, simplified and secure.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
            <Button
              size="lg"
              onClick={handleRegister}
              className="bg-medical-blue hover:bg-medical-blue/90 text-white px-8 py-4 text-lg font-semibold rounded-lg shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105"
            >
              <UserPlus className="mr-2 h-5 w-5" />
              Get Started - Register Now
            </Button>

            <Button
              size="lg"
              variant="outline"
              onClick={handleLogin}
              className="border-2 border-medical-green text-medical-green hover:bg-medical-green hover:text-white px-8 py-4 text-lg font-semibold rounded-lg transition-all duration-300 transform hover:scale-105"
            >
              <Shield className="mr-2 h-5 w-5" />
              Returning Patient? Log In
            </Button>
          </div>
        </div>
      </section>

      {/* 3 Steps Process */}
      <section className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-healthcare-text mb-4">
            3 Simple Steps to Better Healthcare
          </h2>
          <p className="text-lg text-muted-foreground">
            Getting started with our healthcare services is easy and
            straightforward
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <Card className="group hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 border-medical-blue/20">
            <CardContent className="p-8 text-center">
              <div className="bg-medical-blue/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:bg-medical-blue group-hover:text-white transition-all duration-300">
                <UserPlus className="h-8 w-8 text-medical-blue group-hover:text-white" />
              </div>
              <h3 className="text-xl font-bold text-healthcare-text mb-4">
                1. Register
              </h3>
              <p className="text-muted-foreground">
                Create your secure account with basic information. Quick, easy,
                and completely confidential.
              </p>
            </CardContent>
          </Card>

          <Card className="group hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 border-medical-green/20">
            <CardContent className="p-8 text-center">
              <div className="bg-medical-green/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:bg-medical-green group-hover:text-white transition-all duration-300">
                <Users className="h-8 w-8 text-medical-green group-hover:text-white" />
              </div>
              <h3 className="text-xl font-bold text-healthcare-text mb-4">
                2. Complete Profile
              </h3>
              <p className="text-muted-foreground">
                Fill in your medical history and preferences to help us provide
                personalized care.
              </p>
            </CardContent>
          </Card>

          <Card className="group hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 border-medical-blue/20">
            <CardContent className="p-8 text-center">
              <div className="bg-medical-blue/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:bg-medical-blue group-hover:text-white transition-all duration-300">
                <Calendar className="h-8 w-8 text-medical-blue group-hover:text-white" />
              </div>
              <h3 className="text-xl font-bold text-healthcare-text mb-4">
                3. Schedule Appointment
              </h3>
              <p className="text-muted-foreground">
                Book your first appointment with our specialists at your
                convenience.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-white/50 py-16">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-healthcare-text mb-4">
              Why Choose HealthCare Plus?
            </h2>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-6xl mx-auto">
            <div className="flex flex-col items-center text-center group">
              <div className="bg-medical-blue/10 p-4 rounded-full mb-4 group-hover:bg-medical-blue group-hover:text-white transition-all duration-300">
                <Shield className="h-8 w-8 text-medical-blue group-hover:text-white" />
              </div>
              <h3 className="font-semibold text-healthcare-text mb-2">
                Secure & Private
              </h3>
              <p className="text-sm text-muted-foreground">
                HIPAA compliant security
              </p>
            </div>

            <div className="flex flex-col items-center text-center group">
              <div className="bg-medical-green/10 p-4 rounded-full mb-4 group-hover:bg-medical-green group-hover:text-white transition-all duration-300">
                <CheckCircle2 className="h-8 w-8 text-medical-green group-hover:text-white" />
              </div>
              <h3 className="font-semibold text-healthcare-text mb-2">
                Quality Care
              </h3>
              <p className="text-sm text-muted-foreground">
                Certified professionals
              </p>
            </div>

            <div className="flex flex-col items-center text-center group">
              <div className="bg-medical-blue/10 p-4 rounded-full mb-4 group-hover:bg-medical-blue group-hover:text-white transition-all duration-300">
                <Clock className="h-8 w-8 text-medical-blue group-hover:text-white" />
              </div>
              <h3 className="font-semibold text-healthcare-text mb-2">
                24/7 Support
              </h3>
              <p className="text-sm text-muted-foreground">
                Always here for you
              </p>
            </div>

            <div className="flex flex-col items-center text-center group">
              <div className="bg-medical-green/10 p-4 rounded-full mb-4 group-hover:bg-medical-green group-hover:text-white transition-all duration-300">
                <Users className="h-8 w-8 text-medical-green group-hover:text-white" />
              </div>
              <h3 className="font-semibold text-healthcare-text mb-2">
                Personalized
              </h3>
              <p className="text-sm text-muted-foreground">
                Tailored to your needs
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-healthcare-text text-white py-12">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="bg-medical-blue p-2 rounded-lg">
                  <Heart className="h-5 w-5 text-white" />
                </div>
                <span className="text-lg font-bold">HealthCare Plus</span>
              </div>
              <p className="text-gray-300 mb-4">
                Providing exceptional healthcare services with compassion and
                expertise.
              </p>
            </div>

            <div>
              <h3 className="font-semibold mb-4">Contact Info</h3>
              <div className="space-y-2 text-gray-300">
                <div className="flex items-center gap-2">
                  <Phone className="h-4 w-4" />
                  <span>98765 43210</span>
                </div>
                <div className="flex items-center gap-2">
                  <Mail className="h-4 w-4" />
                  <span>info@healthcareplus.com</span>
                </div>
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4" />
                  <span>Alpha Tower, 4th Floor, Bangalore</span>
                </div>
              </div>
            </div>

            <div>
              <h3 className="font-semibold mb-4">Quick Links</h3>
              <ul className="space-y-2 text-gray-300">
                <li>
                  <a
                    href="#"
                    className="hover:text-medical-blue transition-colors"
                  >
                    Services
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-medical-blue transition-colors"
                  >
                    About Us
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-medical-blue transition-colors"
                  >
                    Careers
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-medical-blue transition-colors"
                  >
                    News
                  </a>
                </li>
                <li>
                  <a
                    href="/admin-login"
                    className="hover:text-red-400 transition-colors text-red-300"
                  >
                    Admin Portal
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold mb-4">Legal</h3>
              <ul className="space-y-2 text-gray-300">
                <li>
                  <a
                    href="#"
                    className="hover:text-medical-blue transition-colors"
                  >
                    Privacy Policy
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-medical-blue transition-colors"
                  >
                    Terms of Service
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-medical-blue transition-colors"
                  >
                    HIPAA Notice
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-medical-blue transition-colors"
                  >
                    Accessibility
                  </a>
                </li>
              </ul>
            </div>
          </div>

          <div className="border-t border-gray-600 mt-8 pt-8 text-center text-gray-300">
            <p>&copy; 2025 HealthCare Plus. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
