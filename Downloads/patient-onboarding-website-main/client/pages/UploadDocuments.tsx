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
  Upload,
  FileText,
  CreditCard,
  X,
  Check,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Download,
  Eye,
} from "lucide-react";

interface UploadedFile {
  id: string;
  file: File;
  url: string;
  category: string;
  status: "uploading" | "success" | "error";
}

interface DocumentCategory {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  acceptedTypes: string[];
  maxFiles: number;
  required: boolean;
}

export default function UploadDocuments() {
  const navigate = useNavigate();
  const { updateOnboardingProgress } = useUser();
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [dragActive, setDragActive] = useState<string | null>(null);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRefs = useRef<{ [key: string]: HTMLInputElement | null }>({});

  const documentCategories: DocumentCategory[] = [
    {
      id: "government-id",
      title: "Government ID",
      description: "Driver's license, passport, or state ID",
      icon: FileText,
      acceptedTypes: ["image/jpeg", "image/png", "application/pdf"],
      maxFiles: 2,
      required: true,
    },
    {
      id: "insurance-card",
      title: "Insurance Card",
      description: "Front and back of your insurance card",
      icon: CreditCard,
      acceptedTypes: ["image/jpeg", "image/png", "application/pdf"],
      maxFiles: 2,
      required: true,
    },
    {
      id: "medical-records",
      title: "Medical Records",
      description: "Previous medical records, lab results, etc.",
      icon: Upload,
      acceptedTypes: ["application/pdf", "image/jpeg", "image/png"],
      maxFiles: 5,
      required: false,
    },
  ];

  const handleDragEnter = useCallback(
    (e: React.DragEvent, categoryId: string) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(categoryId);
    },
    [],
  );

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(null);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, categoryId: string) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(null);

    const files = Array.from(e.dataTransfer.files);
    handleFileUpload(files, categoryId);
  }, []);

  const handleFileUpload = async (files: File[], categoryId: string) => {
    const category = documentCategories.find((cat) => cat.id === categoryId);
    if (!category) return;

    const existingFiles = uploadedFiles.filter(
      (file) => file.category === categoryId,
    );
    const remainingSlots = category.maxFiles - existingFiles.length;

    if (remainingSlots <= 0) {
      console.error(
        `Maximum ${category.maxFiles} files allowed for ${category.title}`,
      );
      return;
    }

    const filesToUpload = files.slice(0, remainingSlots);

    for (const file of filesToUpload) {
      // Validate file type
      if (!category.acceptedTypes.includes(file.type)) {
        console.error(
          `Invalid file type for ${category.title}. Accepted: PDF, JPG, PNG`,
        );
        continue;
      }

      // Validate file size (10MB limit)
      if (file.size > 10 * 1024 * 1024) {
        console.error(`File size must be less than 10MB for ${file.name}`);
        continue;
      }

      const fileId = `${categoryId}-${Date.now()}-${Math.random()}`;
      const url = URL.createObjectURL(file);

      const uploadedFile: UploadedFile = {
        id: fileId,
        file,
        url,
        category: categoryId,
        status: "uploading",
      };

      setUploadedFiles((prev) => [...prev, uploadedFile]);

      // Simulate upload process
      setTimeout(
        () => {
          setUploadedFiles((prev) =>
            prev.map((f) =>
              f.id === fileId ? { ...f, status: "success" } : f,
            ),
          );
        },
        1500 + Math.random() * 1000,
      );
    }
  };

  const handleFileSelect = (categoryId: string) => {
    const input = fileInputRefs.current[categoryId];
    if (input) {
      input.click();
    }
  };

  const handleFileInputChange = (
    e: React.ChangeEvent<HTMLInputElement>,
    categoryId: string,
  ) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      handleFileUpload(files, categoryId);
    }
    // Reset input value to allow re-selecting the same file
    e.target.value = "";
  };

  const deleteFile = (fileId: string) => {
    setUploadedFiles((prev) => {
      const fileToDelete = prev.find((f) => f.id === fileId);
      if (fileToDelete) {
        URL.revokeObjectURL(fileToDelete.url);
      }
      return prev.filter((f) => f.id !== fileId);
    });
  };

  const replaceFile = (fileId: string) => {
    const fileToReplace = uploadedFiles.find((f) => f.id === fileId);
    if (fileToReplace) {
      deleteFile(fileId);
      handleFileSelect(fileToReplace.category);
    }
  };

  const getFilesByCategory = (categoryId: string) => {
    return uploadedFiles.filter((file) => file.category === categoryId);
  };

  const getAcceptedTypesString = (types: string[]) => {
    return types
      .map((type) => {
        switch (type) {
          case "application/pdf":
            return "PDF";
          case "image/jpeg":
            return "JPG";
          case "image/png":
            return "PNG";
          default:
            return type;
        }
      })
      .join(", ");
  };

  const canSubmit = () => {
    const requiredCategories = documentCategories.filter((cat) => cat.required);
    return requiredCategories.every((category) => {
      const categoryFiles = getFilesByCategory(category.id);
      return (
        categoryFiles.length > 0 &&
        categoryFiles.every((f) => f.status === "success")
      );
    });
  };

  const handleSubmit = async () => {
    if (!canSubmit()) return;

    setUploading(true);

    // Simulate server upload
    setTimeout(() => {
      setUploading(false);
      updateOnboardingProgress("uploadDocuments", true);
      setShowConfirmation(true);
    }, 2000);
  };

  const handleConfirmationClose = () => {
    setShowConfirmation(false);
    navigate("/profile-setup");
  };

  const renderFilePreview = (file: UploadedFile) => {
    const isImage = file.file.type.startsWith("image/");
    const isPDF = file.file.type === "application/pdf";

    return (
      <div
        key={file.id}
        className="relative bg-white border border-gray-200 rounded-lg p-3 shadow-sm"
      >
        <div className="flex items-start gap-3">
          {/* File Preview */}
          <div className="flex-shrink-0">
            {isImage ? (
              <img
                src={file.url}
                alt={file.file.name}
                className="w-16 h-16 object-cover rounded border"
              />
            ) : isPDF ? (
              <div className="w-16 h-16 bg-red-100 rounded border flex items-center justify-center">
                <FileText className="h-8 w-8 text-red-600" />
              </div>
            ) : (
              <div className="w-16 h-16 bg-gray-100 rounded border flex items-center justify-center">
                <FileText className="h-8 w-8 text-gray-600" />
              </div>
            )}
          </div>

          {/* File Info */}
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">
              {file.file.name}
            </p>
            <p className="text-xs text-gray-500">
              {(file.file.size / 1024 / 1024).toFixed(1)} MB
            </p>

            {/* Status */}
            <div className="mt-1">
              {file.status === "uploading" && (
                <div className="flex items-center gap-1 text-xs text-blue-600">
                  <RefreshCw className="h-3 w-3 animate-spin" />
                  Uploading...
                </div>
              )}
              {file.status === "success" && (
                <div className="flex items-center gap-1 text-xs text-green-600">
                  <CheckCircle2 className="h-3 w-3" />
                  Uploaded
                </div>
              )}
              {file.status === "error" && (
                <div className="flex items-center gap-1 text-xs text-red-600">
                  <AlertCircle className="h-3 w-3" />
                  Error
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex flex-col gap-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => window.open(file.url, "_blank")}
              className="h-6 w-6 p-0"
            >
              <Eye className="h-3 w-3" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => replaceFile(file.id)}
              className="h-6 w-6 p-0"
            >
              <RefreshCw className="h-3 w-3" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => deleteFile(file.id)}
              className="h-6 w-6 p-0 text-red-600 hover:text-red-700"
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </div>
    );
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
              <Upload className="h-8 w-8 text-medical-blue" />
            </div>
            <h1 className="text-3xl font-bold text-healthcare-text mb-2">
              Upload Documents
            </h1>
            <p className="text-lg text-muted-foreground">
              Upload your identification and insurance documents securely
            </p>
          </div>

          {/* Document Categories */}
          <div className="space-y-8">
            {documentCategories.map((category) => {
              const categoryFiles = getFilesByCategory(category.id);
              const isDragActive = dragActive === category.id;

              return (
                <Card
                  key={category.id}
                  className="border-0 shadow-lg bg-white/90 backdrop-blur-sm"
                >
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="bg-medical-blue/10 p-2 rounded-lg">
                          <category.icon className="h-5 w-5 text-medical-blue" />
                        </div>
                        <div>
                          <CardTitle className="text-lg text-healthcare-text flex items-center gap-2">
                            {category.title}
                            {category.required && (
                              <Badge variant="destructive" className="text-xs">
                                Required
                              </Badge>
                            )}
                          </CardTitle>
                          <CardDescription>
                            {category.description}
                          </CardDescription>
                        </div>
                      </div>
                      <div className="text-right text-sm text-muted-foreground">
                        {categoryFiles.length} / {category.maxFiles} files
                      </div>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    {/* Upload Area */}
                    <div
                      className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 cursor-pointer ${
                        isDragActive
                          ? "border-medical-blue bg-medical-light-blue/20"
                          : categoryFiles.length >= category.maxFiles
                            ? "border-gray-300 bg-gray-50 cursor-not-allowed"
                            : "border-gray-300 hover:border-medical-blue hover:bg-medical-light-blue/10"
                      }`}
                      onDragEnter={(e) => handleDragEnter(e, category.id)}
                      onDragLeave={handleDragLeave}
                      onDragOver={handleDragOver}
                      onDrop={(e) => handleDrop(e, category.id)}
                      onClick={() =>
                        categoryFiles.length < category.maxFiles &&
                        handleFileSelect(category.id)
                      }
                    >
                      <input
                        ref={(el) => (fileInputRefs.current[category.id] = el)}
                        type="file"
                        multiple
                        accept={category.acceptedTypes.join(",")}
                        onChange={(e) => handleFileInputChange(e, category.id)}
                        className="hidden"
                      />

                      {categoryFiles.length >= category.maxFiles ? (
                        <div className="text-gray-500">
                          <Upload className="h-8 w-8 mx-auto mb-2 opacity-50" />
                          <p className="font-medium">Maximum files reached</p>
                          <p className="text-sm">
                            You can replace existing files using the replace
                            button
                          </p>
                        </div>
                      ) : (
                        <div className="text-medical-blue">
                          <Upload className="h-8 w-8 mx-auto mb-2" />
                          <p className="font-medium">
                            {isDragActive
                              ? "Drop files here"
                              : "Drag and drop files here, or click to browse"}
                          </p>
                          <p className="text-sm text-muted-foreground mt-1">
                            Supports:{" "}
                            {getAcceptedTypesString(category.acceptedTypes)}{" "}
                            (Max 10MB each)
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Uploaded Files */}
                    {categoryFiles.length > 0 && (
                      <div className="space-y-3">
                        <h4 className="font-medium text-healthcare-text">
                          Uploaded Files:
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {categoryFiles.map(renderFilePreview)}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* Submit Button */}
          <div className="mt-8 text-center">
            <Button
              onClick={handleSubmit}
              disabled={!canSubmit() || uploading}
              className="bg-medical-green hover:bg-medical-green/90 text-white px-8 py-3 text-lg font-semibold"
            >
              {uploading ? (
                <div className="flex items-center gap-2">
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  Uploading Documents...
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Check className="h-4 w-4" />
                  Submit Documents
                </div>
              )}
            </Button>

            {!canSubmit() && (
              <p className="text-sm text-muted-foreground mt-2">
                Please upload all required documents before submitting
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Confirmation Modal */}
      <Dialog open={showConfirmation} onOpenChange={setShowConfirmation}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader className="text-center">
            <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle2 className="h-8 w-8 text-green-600" />
            </div>
            <DialogTitle className="text-xl font-bold text-healthcare-text text-center">
              Documents Successfully Uploaded!
            </DialogTitle>
            <DialogDescription className="text-base text-center">
              Your documents have been securely uploaded and are now being
              reviewed. You'll receive a notification once the verification is
              complete.
            </DialogDescription>
          </DialogHeader>

          <div className="flex justify-center gap-3 mt-6">
            <Button
              onClick={handleConfirmationClose}
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
