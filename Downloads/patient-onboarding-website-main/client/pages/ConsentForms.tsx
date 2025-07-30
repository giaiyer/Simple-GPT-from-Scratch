import { useState, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "@/lib/UserContext";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Heart,
  ArrowLeft,
  FileText,
  Shield,
  CheckCircle2,
  PenTool,
  RotateCcw,
  Download,
  Eye,
  Signature,
} from "lucide-react";

interface ConsentForm {
  id: string;
  title: string;
  description: string;
  category: string;
  required: boolean;
  content: {
    sections: {
      title: string;
      content: string;
    }[];
  };
  status: "not-started" | "read" | "signed";
  signature?: string;
  signedDate?: Date;
}

interface SignaturePadProps {
  onSave: (signature: string) => void;
  onClear: () => void;
}

// Simple signature pad component
const SignaturePad: React.FC<SignaturePadProps> = ({ onSave, onClear }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isEmpty, setIsEmpty] = useState(true);

  const startDrawing = useCallback(
    (event: React.MouseEvent | React.TouchEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      setIsDrawing(true);
      setIsEmpty(false);

      const rect = canvas.getBoundingClientRect();
      const clientX =
        "touches" in event ? event.touches[0].clientX : event.clientX;
      const clientY =
        "touches" in event ? event.touches[0].clientY : event.clientY;

      const x = clientX - rect.left;
      const y = clientY - rect.top;

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.beginPath();
        ctx.moveTo(x, y);
      }
    },
    [],
  );

  const draw = useCallback(
    (event: React.MouseEvent | React.TouchEvent) => {
      if (!isDrawing) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const clientX =
        "touches" in event ? event.touches[0].clientX : event.clientX;
      const clientY =
        "touches" in event ? event.touches[0].clientY : event.clientY;

      const x = clientX - rect.left;
      const y = clientY - rect.top;

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    },
    [isDrawing],
  );

  const stopDrawing = useCallback(() => {
    setIsDrawing(false);
  }, []);

  const clearSignature = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      setIsEmpty(true);
      onClear();
    }
  }, [onClear]);

  const saveSignature = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || isEmpty) return;

    const dataURL = canvas.toDataURL();
    onSave(dataURL);
  }, [onSave, isEmpty]);

  // Initialize canvas
  const initCanvas = useCallback((canvas: HTMLCanvasElement | null) => {
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
    }
  }, []);

  return (
    <div className="space-y-4">
      <div className="border border-gray-300 rounded-lg bg-white">
        <canvas
          ref={(el) => {
            canvasRef.current = el;
            initCanvas(el);
          }}
          width={400}
          height={200}
          className="w-full h-48 cursor-crosshair"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
        />
      </div>
      <div className="flex gap-2">
        <Button variant="outline" onClick={clearSignature} className="flex-1">
          <RotateCcw className="mr-2 h-4 w-4" />
          Clear
        </Button>
        <Button
          onClick={saveSignature}
          disabled={isEmpty}
          className="flex-1 bg-medical-blue hover:bg-medical-blue/90"
        >
          <CheckCircle2 className="mr-2 h-4 w-4" />
          Save Signature
        </Button>
      </div>
    </div>
  );
};

export default function ConsentForms() {
  const navigate = useNavigate();
  const { updateOnboardingProgress } = useUser();
  const [forms, setForms] = useState<ConsentForm[]>([
    {
      id: "hipaa-agreement",
      title: "HIPAA Privacy Agreement",
      description: "Authorization for use and disclosure of health information",
      category: "Privacy",
      required: true,
      content: {
        sections: [
          {
            title: "Understanding Your Health Information Privacy Rights",
            content: `This notice describes how medical information about you may be used and disclosed and how you can get access to this information. Please review it carefully.

Your health information is personal and private. We are committed to protecting your health information. We create a record of the care and services you receive at our practice. We need this record to provide you with quality care and to comply with certain legal requirements.

This notice applies to all of the records of your care generated by this practice, whether made by practice personnel or your personal doctor. Your personal doctor may have different policies or notices regarding the doctor's use and disclosure of your medical information created in the doctor's office or clinic.`,
          },
          {
            title: "How We May Use and Disclose Medical Information About You",
            content: `We may use and disclose your medical information for the following purposes:

For Treatment: We may use medical information about you to provide you with medical treatment or services. We may disclose medical information about you to doctors, nurses, technicians, medical students, or other personnel who are involved in taking care of you.

For Payment: We may use and disclose medical information about you so that the treatment and services you receive may be billed to and payment may be collected from you, an insurance company, or a third party.

For Health Care Operations: We may use and disclose medical information about you for health care operations. These uses and disclosures are necessary to run the practice and make sure that all of our patients receive quality care.`,
          },
          {
            title: "Your Rights Regarding Medical Information About You",
            content: `You have the following rights regarding medical information we maintain about you:

Right to Inspect and Copy: You have the right to inspect and copy medical information that may be used to make decisions about your care.

Right to Amend: If you feel that medical information we have about you is incorrect or incomplete, you may ask us to amend the information.

Right to an Accounting of Disclosures: You have the right to request an "accounting of disclosures."

Right to Request Restrictions: You have the right to request a restriction or limitation on the medical information we use or disclose about you for treatment, payment, or health care operations.`,
          },
        ],
      },
      status: "not-started",
    },
    {
      id: "treatment-consent",
      title: "General Treatment Consent",
      description: "Consent for examination, diagnosis, and treatment",
      category: "Treatment",
      required: true,
      content: {
        sections: [
          {
            title: "Consent to Treatment",
            content: `I voluntarily consent to such care, treatment, and services as may be deemed necessary by the attending physician(s) and other healthcare providers involved in my care. This includes, but is not limited to, diagnostic procedures, medical treatment, and nursing care.

I understand that the practice of medicine is not an exact science and that no guarantee or assurance has been made to me as to the result of any examination or treatment. I acknowledge that no guarantee of success has been made to me regarding the proposed treatment.`,
          },
          {
            title: "Risks and Benefits",
            content: `I understand that all medical and surgical procedures involve risks and benefits, and that complications may occur as a result of the treatment, including but not limited to:
            
• Infection
• Bleeding
• Blood clots
• Adverse reactions to medications
• The need for additional procedures
• Permanent disability
• Death

I understand that I have the right to be informed of such risks and benefits, and of alternative forms of treatment, and I may withdraw my consent at any time.`,
          },
          {
            title: "Financial Responsibility",
            content: `I understand that I am financially responsible for all charges whether or not paid by insurance. I hereby authorize payment of medical benefits to this practice for services rendered. I authorize the use of this signature on all insurance submissions.

I understand that if I fail to keep an appointment, I may be charged for the reserved time. I agree to pay all costs of collection and reasonable attorney fees in the event that collection efforts are required.`,
          },
        ],
      },
      status: "not-started",
    },
    {
      id: "financial-responsibility",
      title: "Financial Responsibility Agreement",
      description: "Understanding of payment policies and insurance coverage",
      category: "Financial",
      required: true,
      content: {
        sections: [
          {
            title: "Payment Policies",
            content: `Payment is due at the time services are rendered unless other arrangements have been made in advance. We accept cash, check, and major credit cards.

If you have insurance, we will bill your insurance company as a courtesy. However, you are ultimately responsible for all charges incurred. If your insurance company has not paid your account in full within 60 days, the balance will automatically become your responsibility.`,
          },
          {
            title: "Insurance Verification",
            content: `It is your responsibility to know your insurance benefits. We will verify your insurance as a courtesy, but this verification is not a guarantee of payment by your insurance company.

You are responsible for obtaining any required referrals or authorizations from your primary care physician or insurance company before receiving services. Failure to obtain required referrals may result in denial of coverage by your insurance company.`,
          },
        ],
      },
      status: "not-started",
    },
    {
      id: "emergency-contact-consent",
      title: "Emergency Contact Authorization",
      description: "Permission to contact designated emergency contacts",
      category: "Emergency",
      required: false,
      content: {
        sections: [
          {
            title: "Emergency Contact Authorization",
            content: `I authorize HealthCare Plus to contact my designated emergency contact(s) in the event of a medical emergency when I am unable to make decisions for myself.

I understand that this authorization allows my emergency contacts to receive information about my medical condition and treatment only in emergency situations where immediate decisions are necessary for my care.`,
          },
        ],
      },
      status: "not-started",
    },
    {
      id: "telehealth-consent",
      title: "Telehealth Services Consent",
      description: "Agreement to participate in telehealth appointments",
      category: "Technology",
      required: false,
      content: {
        sections: [
          {
            title: "Telehealth Services",
            content: `I understand that telehealth involves the use of electronic communications to enable healthcare providers at different locations to share individual patient medical information for the purpose of improving patient care.

I understand that the laws that protect privacy and the confidentiality of medical information also apply to telehealth, and that no information obtained in the use of telehealth will be disclosed to researchers or other entities without my consent.`,
          },
          {
            title: "Technical Requirements and Limitations",
            content: `I understand that telehealth may involve electronic communication of my personal medical information to other medical practitioners who may be located in other areas, including out of state.

I understand that it is my duty to inform my healthcare provider of electronic interactions regarding my care that I may have with other healthcare providers.

I understand that I may withhold or withdraw my consent to telehealth at any time without affecting my right to future care or treatment.`,
          },
        ],
      },
      status: "not-started",
    },
  ]);

  const [currentSigningForm, setCurrentSigningForm] = useState<string | null>(
    null,
  );
  const [showFullForm, setShowFullForm] = useState<string | null>(null);
  const [showCompletion, setShowCompletion] = useState(false);

  const completedForms = forms.filter(
    (form) => form.status === "signed",
  ).length;
  const requiredForms = forms.filter((form) => form.required).length;
  const requiredCompletedForms = forms.filter(
    (form) => form.required && form.status === "signed",
  ).length;
  const totalForms = forms.length;
  const progressPercentage = Math.round((completedForms / totalForms) * 100);

  const markFormAsRead = (formId: string) => {
    setForms((prev) =>
      prev.map((form) =>
        form.id === formId && form.status === "not-started"
          ? { ...form, status: "read" as const }
          : form,
      ),
    );
  };

  const handleSignature = (formId: string, signature: string) => {
    setForms((prev) =>
      prev.map((form) =>
        form.id === formId
          ? {
              ...form,
              status: "signed" as const,
              signature,
              signedDate: new Date(),
            }
          : form,
      ),
    );
    setCurrentSigningForm(null);
  };

  const canSubmit = () => {
    return requiredCompletedForms === requiredForms;
  };

  const handleSubmit = () => {
    if (!canSubmit()) return;
    updateOnboardingProgress("consentForms", true);
    setShowCompletion(true);
  };

  const handleCompletionClose = () => {
    setShowCompletion(false);
    navigate("/profile-setup");
  };

  const getStatusBadge = (status: ConsentForm["status"]) => {
    switch (status) {
      case "signed":
        return <Badge className="bg-green-100 text-green-800">Signed</Badge>;
      case "read":
        return <Badge className="bg-blue-100 text-blue-800">Read</Badge>;
      case "not-started":
        return <Badge className="bg-gray-100 text-gray-600">Not Started</Badge>;
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
              <FileText className="h-8 w-8 text-medical-blue" />
            </div>
            <h1 className="text-3xl font-bold text-healthcare-text mb-2">
              Consent Forms
            </h1>
            <p className="text-lg text-muted-foreground">
              Review and sign the necessary medical consent documents
            </p>
          </div>

          {/* Progress Tracker */}
          <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm mb-8">
            <CardHeader>
              <CardTitle className="text-xl text-healthcare-text">
                Forms Progress
              </CardTitle>
              <CardDescription>
                {completedForms} of {totalForms} forms completed
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Progress value={progressPercentage} className="h-3" />
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
                <div className="bg-medical-light-blue/20 p-3 rounded-lg">
                  <div className="text-xl font-bold text-medical-blue">
                    {completedForms}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Total Signed
                  </div>
                </div>
                <div className="bg-medical-light-green/20 p-3 rounded-lg">
                  <div className="text-xl font-bold text-medical-green">
                    {requiredCompletedForms} / {requiredForms}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Required Forms
                  </div>
                </div>
                <div className="bg-orange-100 p-3 rounded-lg">
                  <div className="text-xl font-bold text-orange-600">
                    {progressPercentage}%
                  </div>
                  <div className="text-sm text-muted-foreground">Complete</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Consent Forms Accordion */}
          <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-xl text-healthcare-text">
                Digital Consent Forms
              </CardTitle>
              <CardDescription>
                Click to expand each form, read carefully, and provide your
                digital signature
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Accordion type="single" collapsible className="space-y-4">
                {forms.map((form, index) => (
                  <AccordionItem
                    key={form.id}
                    value={form.id}
                    className={`border rounded-lg px-4 ${
                      form.status === "signed"
                        ? "border-green-300 bg-green-50/50"
                        : form.status === "read"
                          ? "border-blue-300 bg-blue-50/50"
                          : "border-gray-300"
                    }`}
                  >
                    <AccordionTrigger className="hover:no-underline">
                      <div className="flex items-center justify-between w-full pr-4">
                        <div className="flex items-center gap-4 text-left">
                          <div className="bg-medical-blue/10 p-2 rounded-full">
                            {form.category === "Privacy" && (
                              <Shield className="h-4 w-4 text-medical-blue" />
                            )}
                            {form.category === "Treatment" && (
                              <FileText className="h-4 w-4 text-medical-blue" />
                            )}
                            {form.category === "Financial" && (
                              <FileText className="h-4 w-4 text-medical-blue" />
                            )}
                            {form.category === "Emergency" && (
                              <FileText className="h-4 w-4 text-medical-blue" />
                            )}
                            {form.category === "Technology" && (
                              <FileText className="h-4 w-4 text-medical-blue" />
                            )}
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <h3 className="font-semibold text-healthcare-text">
                                {form.title}
                              </h3>
                              {form.required && (
                                <Badge
                                  variant="destructive"
                                  className="text-xs"
                                >
                                  Required
                                </Badge>
                              )}
                            </div>
                            <p className="text-sm text-muted-foreground">
                              {form.description}
                            </p>
                            <div className="mt-1">
                              {getStatusBadge(form.status)}
                            </div>
                          </div>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Form {index + 1} of {totalForms}
                        </div>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="pt-4">
                      <div className="space-y-4">
                        {/* Form Content Preview */}
                        <div className="bg-gray-50 p-4 rounded-lg">
                          <h4 className="font-medium text-healthcare-text mb-2">
                            {form.content.sections[0].title}
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            {form.content.sections[0].content.substring(0, 200)}
                            ...
                          </p>
                        </div>

                        {/* Action Buttons */}
                        <div className="flex gap-3">
                          <Button
                            variant="outline"
                            onClick={() => {
                              setShowFullForm(form.id);
                              markFormAsRead(form.id);
                            }}
                            className="flex-1 border-medical-blue text-medical-blue hover:bg-medical-blue hover:text-white"
                          >
                            <Eye className="mr-2 h-4 w-4" />
                            Read Full Form
                          </Button>

                          {form.status !== "not-started" && (
                            <Button
                              onClick={() => setCurrentSigningForm(form.id)}
                              disabled={form.status === "signed"}
                              className="flex-1 bg-medical-green hover:bg-medical-green/90"
                            >
                              {form.status === "signed" ? (
                                <>
                                  <CheckCircle2 className="mr-2 h-4 w-4" />
                                  Signed
                                </>
                              ) : (
                                <>
                                  <PenTool className="mr-2 h-4 w-4" />
                                  Sign Form
                                </>
                              )}
                            </Button>
                          )}
                        </div>

                        {/* Signature Info */}
                        {form.status === "signed" && form.signedDate && (
                          <div className="bg-green-50 p-3 rounded-lg border border-green-200">
                            <div className="flex items-center gap-2 text-green-800">
                              <CheckCircle2 className="h-4 w-4" />
                              <span className="text-sm font-medium">
                                Signed on {form.signedDate.toLocaleDateString()}{" "}
                                at {form.signedDate.toLocaleTimeString()}
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            </CardContent>
          </Card>

          {/* Submit Button */}
          <div className="mt-8 text-center">
            <Button
              onClick={handleSubmit}
              disabled={!canSubmit()}
              className="bg-medical-blue hover:bg-medical-blue/90 text-white px-8 py-3 text-lg font-semibold"
            >
              <CheckCircle2 className="mr-2 h-5 w-5" />
              Submit All Forms
            </Button>

            {!canSubmit() && (
              <p className="text-sm text-muted-foreground mt-2">
                Please complete all required forms before submitting
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Full Form Modal */}
      <Dialog open={!!showFullForm} onOpenChange={() => setShowFullForm(null)}>
        <DialogContent className="sm:max-w-4xl max-h-[80vh] overflow-y-auto">
          {showFullForm && (
            <>
              <DialogHeader>
                <DialogTitle className="text-xl font-bold text-healthcare-text">
                  {forms.find((f) => f.id === showFullForm)?.title}
                </DialogTitle>
                <DialogDescription>
                  Please read this form carefully before signing
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-6">
                {forms
                  .find((f) => f.id === showFullForm)
                  ?.content.sections.map((section, index) => (
                    <div key={index} className="space-y-3">
                      <h3 className="text-lg font-semibold text-healthcare-text">
                        {section.title}
                      </h3>
                      <div className="text-sm text-muted-foreground whitespace-pre-line leading-relaxed">
                        {section.content}
                      </div>
                    </div>
                  ))}
              </div>

              <div className="flex justify-end gap-3 pt-6 border-t">
                <Button variant="outline" onClick={() => setShowFullForm(null)}>
                  Close
                </Button>
                <Button
                  onClick={() => {
                    setShowFullForm(null);
                    setCurrentSigningForm(showFullForm);
                  }}
                  className="bg-medical-green hover:bg-medical-green/90"
                >
                  <PenTool className="mr-2 h-4 w-4" />
                  Proceed to Sign
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* Signature Modal */}
      <Dialog
        open={!!currentSigningForm}
        onOpenChange={() => setCurrentSigningForm(null)}
      >
        <DialogContent className="sm:max-w-lg">
          {currentSigningForm && (
            <>
              <DialogHeader>
                <DialogTitle className="text-xl font-bold text-healthcare-text flex items-center gap-2">
                  <Signature className="h-5 w-5" />
                  Digital Signature
                </DialogTitle>
                <DialogDescription>
                  Please provide your digital signature for:{" "}
                  {forms.find((f) => f.id === currentSigningForm)?.title}
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-4">
                <div className="text-sm text-muted-foreground bg-blue-50 p-3 rounded-lg">
                  By signing below, you acknowledge that you have read and
                  understood the form content and agree to its terms.
                </div>

                <SignaturePad
                  onSave={(signature) =>
                    handleSignature(currentSigningForm, signature)
                  }
                  onClear={() => {}}
                />
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* Completion Modal */}
      <Dialog open={showCompletion} onOpenChange={setShowCompletion}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader className="text-center">
            <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle2 className="h-8 w-8 text-green-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              Forms Successfully Submitted!
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              All consent forms have been completed and digitally signed. Your
              information is now secure in our system.
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
