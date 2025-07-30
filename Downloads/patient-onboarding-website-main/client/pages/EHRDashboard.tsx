import { useState, useEffect } from "react";
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
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Heart,
  ArrowLeft,
  Settings,
  Server,
  Shield,
  Activity,
  Database,
  Key,
  CheckCircle2,
  AlertTriangle,
  Clock,
  TrendingUp,
  Users,
  FileText,
  Download,
  Plus,
  Edit,
  Trash2,
  Eye,
  EyeOff,
  RefreshCw,
  AlertCircle,
  Zap,
  BarChart3,
  Calendar,
  Bell,
} from "lucide-react";
import AppHeader from "@/components/AppHeader";

interface EHRSystem {
  id: string;
  name: string;
  version: string;
  status: "connected" | "disconnected" | "error" | "syncing";
  lastSync: string;
  apiKey: string;
  endpoint: string;
  recordsCount: number;
  compliance: "compliant" | "warning" | "critical";
  uptime: number;
}

interface DataSync {
  id: string;
  system: string;
  type: "patient" | "appointment" | "medication" | "lab" | "imaging";
  status: "completed" | "failed" | "in_progress" | "scheduled";
  timestamp: string;
  recordsProcessed: number;
  errors: number;
}

interface ComplianceAlert {
  id: string;
  type: "HIPAA" | "HL7" | "FHIR" | "API_LIMIT" | "SECURITY";
  severity: "low" | "medium" | "high" | "critical";
  message: string;
  timestamp: string;
  resolved: boolean;
}

export default function EHRDashboard() {
  const navigate = useNavigate();
  const { isAdmin, userData } = useUser();
  const [showAddEHR, setShowAddEHR] = useState(false);
  const [showAPIKey, setShowAPIKey] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState("overview");

  // Admin access control
  useEffect(() => {
    if (!isAdmin()) {
      // Redirect non-admin users to admin login
      navigate("/admin-login");
    }
  }, [isAdmin, navigate]);

  // Show loading if not admin (will redirect)
  if (!isAdmin()) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-medical-light-blue via-background to-medical-light-green flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-medical-blue mx-auto mb-4"></div>
          <p className="text-muted-foreground">Checking authorization...</p>
        </div>
      </div>
    );
  }

  // Mock data for EHR systems
  const [ehrSystems, setEHRSystems] = useState<EHRSystem[]>([
    {
      id: "epic-1",
      name: "Epic MyChart",
      version: "2023.1",
      status: "connected",
      lastSync: "2024-01-15T10:30:00Z",
      apiKey: "epic_prod_ak_7x9z2m5n8q1w3e4r",
      endpoint: "https://fhir.epic.com/interconnect-fhir-oauth",
      recordsCount: 12547,
      compliance: "compliant",
      uptime: 99.8,
    },
    {
      id: "cerner-1",
      name: "Cerner PowerChart",
      version: "2024.02",
      status: "syncing",
      lastSync: "2024-01-15T09:45:00Z",
      apiKey: "cerner_api_key_a1b2c3d4e5f6g7h8",
      endpoint: "https://fhir-open.cerner.com/r4",
      recordsCount: 8934,
      compliance: "warning",
      uptime: 98.2,
    },
    {
      id: "allscripts-1",
      name: "Allscripts TouchWorks",
      version: "19.0",
      status: "error",
      lastSync: "2024-01-14T16:20:00Z",
      apiKey: "as_touchworks_key_xyz789",
      endpoint: "https://developer.allscripts.com/fhir",
      recordsCount: 5621,
      compliance: "critical",
      uptime: 87.5,
    },
    {
      id: "athena-1",
      name: "athenahealth",
      version: "23.3",
      status: "connected",
      lastSync: "2024-01-15T11:15:00Z",
      apiKey: "athena_v1_token_mn8op9qr",
      endpoint: "https://api.athenahealth.com/fhir/r4",
      recordsCount: 3456,
      compliance: "compliant",
      uptime: 99.1,
    },
  ]);

  // Mock data for recent syncs
  const [dataSyncs] = useState<DataSync[]>([
    {
      id: "sync-1",
      system: "Epic MyChart",
      type: "patient",
      status: "completed",
      timestamp: "2024-01-15T10:30:00Z",
      recordsProcessed: 245,
      errors: 0,
    },
    {
      id: "sync-2",
      system: "Cerner PowerChart",
      type: "appointment",
      status: "in_progress",
      timestamp: "2024-01-15T10:25:00Z",
      recordsProcessed: 89,
      errors: 2,
    },
    {
      id: "sync-3",
      system: "athenahealth",
      type: "lab",
      status: "completed",
      timestamp: "2024-01-15T10:15:00Z",
      recordsProcessed: 156,
      errors: 0,
    },
    {
      id: "sync-4",
      system: "Allscripts TouchWorks",
      type: "medication",
      status: "failed",
      timestamp: "2024-01-15T09:45:00Z",
      recordsProcessed: 0,
      errors: 15,
    },
  ]);

  // Mock compliance alerts
  const [complianceAlerts] = useState<ComplianceAlert[]>([
    {
      id: "alert-1",
      type: "HIPAA",
      severity: "high",
      message: "Unauthorized access attempt detected on Epic integration",
      timestamp: "2024-01-15T09:30:00Z",
      resolved: false,
    },
    {
      id: "alert-2",
      type: "API_LIMIT",
      severity: "medium",
      message: "Cerner API rate limit approaching (85% of daily quota)",
      timestamp: "2024-01-15T08:45:00Z",
      resolved: false,
    },
    {
      id: "alert-3",
      type: "FHIR",
      severity: "low",
      message: "Non-critical FHIR validation warnings in Allscripts sync",
      timestamp: "2024-01-15T07:20:00Z",
      resolved: true,
    },
  ]);

  const getStatusBadge = (status: EHRSystem["status"]) => {
    switch (status) {
      case "connected":
        return <Badge className="bg-green-100 text-green-800">Connected</Badge>;
      case "disconnected":
        return (
          <Badge className="bg-gray-100 text-gray-800">Disconnected</Badge>
        );
      case "error":
        return <Badge className="bg-red-100 text-red-800">Error</Badge>;
      case "syncing":
        return <Badge className="bg-blue-100 text-blue-800">Syncing</Badge>;
    }
  };

  const getComplianceBadge = (compliance: EHRSystem["compliance"]) => {
    switch (compliance) {
      case "compliant":
        return <Badge className="bg-green-100 text-green-800">Compliant</Badge>;
      case "warning":
        return <Badge className="bg-yellow-100 text-yellow-800">Warning</Badge>;
      case "critical":
        return <Badge className="bg-red-100 text-red-800">Critical</Badge>;
    }
  };

  const getSyncStatusBadge = (status: DataSync["status"]) => {
    switch (status) {
      case "completed":
        return <Badge className="bg-green-100 text-green-800">Completed</Badge>;
      case "failed":
        return <Badge className="bg-red-100 text-red-800">Failed</Badge>;
      case "in_progress":
        return <Badge className="bg-blue-100 text-blue-800">In Progress</Badge>;
      case "scheduled":
        return <Badge className="bg-gray-100 text-gray-800">Scheduled</Badge>;
    }
  };

  const getSeverityBadge = (severity: ComplianceAlert["severity"]) => {
    switch (severity) {
      case "low":
        return <Badge className="bg-blue-100 text-blue-800">Low</Badge>;
      case "medium":
        return <Badge className="bg-yellow-100 text-yellow-800">Medium</Badge>;
      case "high":
        return <Badge className="bg-orange-100 text-orange-800">High</Badge>;
      case "critical":
        return <Badge className="bg-red-100 text-red-800">Critical</Badge>;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const connectedSystems = ehrSystems.filter(
    (sys) => sys.status === "connected",
  ).length;
  const totalRecords = ehrSystems.reduce(
    (sum, sys) => sum + sys.recordsCount,
    0,
  );
  const avgUptime =
    ehrSystems.reduce((sum, sys) => sum + sys.uptime, 0) / ehrSystems.length;
  const activeAlerts = complianceAlerts.filter(
    (alert) => !alert.resolved,
  ).length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light-blue via-background to-medical-light-green">
      <AppHeader adminMode={true} />

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          {/* Page Header */}
          <div className="text-center mb-8">
            <div className="bg-medical-blue/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Server className="h-8 w-8 text-medical-blue" />
            </div>
            <h1 className="text-3xl font-bold text-healthcare-text mb-2">
              EHR Integration Dashboard
            </h1>
            <p className="text-lg text-muted-foreground">
              Manage connections with Electronic Health Record systems and
              monitor data flow
            </p>
          </div>

          {/* Overview Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">
                      Connected Systems
                    </p>
                    <p className="text-2xl font-bold text-medical-blue">
                      {connectedSystems}/{ehrSystems.length}
                    </p>
                  </div>
                  <div className="bg-medical-blue/10 p-3 rounded-full">
                    <Database className="h-6 w-6 text-medical-blue" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">
                      Total Records
                    </p>
                    <p className="text-2xl font-bold text-medical-green">
                      {totalRecords.toLocaleString()}
                    </p>
                  </div>
                  <div className="bg-medical-green/10 p-3 rounded-full">
                    <FileText className="h-6 w-6 text-medical-green" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">
                      Average Uptime
                    </p>
                    <p className="text-2xl font-bold text-green-600">
                      {avgUptime.toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-green-100 p-3 rounded-full">
                    <TrendingUp className="h-6 w-6 text-green-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">
                      Active Alerts
                    </p>
                    <p className="text-2xl font-bold text-orange-600">
                      {activeAlerts}
                    </p>
                  </div>
                  <div className="bg-orange-100 p-3 rounded-full">
                    <AlertTriangle className="h-6 w-6 text-orange-600" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main Content Tabs */}
          <Tabs
            value={selectedTab}
            onValueChange={setSelectedTab}
            className="space-y-6"
          >
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="overview">Systems Overview</TabsTrigger>
              <TabsTrigger value="sync">Data Sync</TabsTrigger>
              <TabsTrigger value="compliance">Compliance</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
            </TabsList>

            {/* Systems Overview Tab */}
            <TabsContent value="overview" className="space-y-6">
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-healthcare-text">
                  EHR Systems
                </h2>
                <Dialog open={showAddEHR} onOpenChange={setShowAddEHR}>
                  <DialogTrigger asChild>
                    <Button className="bg-medical-blue hover:bg-medical-blue/90">
                      <Plus className="mr-2 h-4 w-4" />
                      Add EHR System
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                      <DialogTitle>Add New EHR System</DialogTitle>
                      <DialogDescription>
                        Configure a new Electronic Health Record system
                        integration
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="ehr-name">EHR System</Label>
                        <Select>
                          <SelectTrigger>
                            <SelectValue placeholder="Select EHR system" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="epic">Epic</SelectItem>
                            <SelectItem value="cerner">Cerner</SelectItem>
                            <SelectItem value="allscripts">
                              Allscripts
                            </SelectItem>
                            <SelectItem value="athena">athenahealth</SelectItem>
                            <SelectItem value="nextgen">NextGen</SelectItem>
                            <SelectItem value="eclinicalworks">
                              eClinicalWorks
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="endpoint">FHIR Endpoint</Label>
                        <Input
                          id="endpoint"
                          placeholder="https://fhir.example.com/r4"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="api-key">API Key</Label>
                        <Input
                          id="api-key"
                          type="password"
                          placeholder="Enter API key"
                        />
                      </div>
                      <div className="flex gap-3">
                        <Button
                          variant="outline"
                          onClick={() => setShowAddEHR(false)}
                          className="flex-1"
                        >
                          Cancel
                        </Button>
                        <Button className="flex-1 bg-medical-green hover:bg-medical-green/90">
                          Test & Save
                        </Button>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>

              <div className="grid gap-6">
                {ehrSystems.map((system) => (
                  <Card
                    key={system.id}
                    className="border-0 shadow-lg bg-white/90 backdrop-blur-sm"
                  >
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-4">
                          <div className="bg-medical-blue/10 p-3 rounded-full">
                            <Server className="h-6 w-6 text-medical-blue" />
                          </div>
                          <div>
                            <h3 className="text-lg font-semibold text-healthcare-text">
                              {system.name}
                            </h3>
                            <p className="text-sm text-muted-foreground">
                              Version {system.version}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          {getStatusBadge(system.status)}
                          {getComplianceBadge(system.compliance)}
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">
                            Records Count
                          </p>
                          <p className="text-lg font-semibold text-healthcare-text">
                            {system.recordsCount.toLocaleString()}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">
                            Last Sync
                          </p>
                          <p className="text-sm text-healthcare-text">
                            {formatTimestamp(system.lastSync)}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">
                            Uptime
                          </p>
                          <p className="text-lg font-semibold text-green-600">
                            {system.uptime}%
                          </p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">
                            API Key
                          </p>
                          <div className="flex items-center gap-2">
                            <Input
                              type={
                                showAPIKey === system.id ? "text" : "password"
                              }
                              value={system.apiKey}
                              readOnly
                              className="text-xs h-8"
                            />
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() =>
                                setShowAPIKey(
                                  showAPIKey === system.id ? null : system.id,
                                )
                              }
                            >
                              {showAPIKey === system.id ? (
                                <EyeOff className="h-4 w-4" />
                              ) : (
                                <Eye className="h-4 w-4" />
                              )}
                            </Button>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <Button variant="outline" size="sm">
                          <RefreshCw className="mr-2 h-4 w-4" />
                          Sync Now
                        </Button>
                        <Button variant="outline" size="sm">
                          <Settings className="mr-2 h-4 w-4" />
                          Configure
                        </Button>
                        <Button variant="outline" size="sm">
                          <BarChart3 className="mr-2 h-4 w-4" />
                          View Logs
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            {/* Data Sync Tab */}
            <TabsContent value="sync" className="space-y-6">
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-healthcare-text">
                  Data Synchronization
                </h2>
                <Button className="bg-medical-green hover:bg-medical-green/90">
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Sync All Systems
                </Button>
              </div>

              <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle>Recent Sync Activities</CardTitle>
                  <CardDescription>
                    Real-time monitoring of data synchronization processes
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>System</TableHead>
                        <TableHead>Data Type</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Timestamp</TableHead>
                        <TableHead>Records</TableHead>
                        <TableHead>Errors</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {dataSyncs.map((sync) => (
                        <TableRow key={sync.id}>
                          <TableCell className="font-medium">
                            {sync.system}
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className="capitalize">
                              {sync.type}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            {getSyncStatusBadge(sync.status)}
                          </TableCell>
                          <TableCell className="text-sm">
                            {formatTimestamp(sync.timestamp)}
                          </TableCell>
                          <TableCell>{sync.recordsProcessed}</TableCell>
                          <TableCell>
                            {sync.errors > 0 ? (
                              <span className="text-red-600 font-medium">
                                {sync.errors}
                              </span>
                            ) : (
                              <span className="text-green-600">0</span>
                            )}
                          </TableCell>
                          <TableCell>
                            <Button variant="ghost" size="sm">
                              <Eye className="h-4 w-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              {/* Sync Schedule */}
              <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle>Sync Schedule Configuration</CardTitle>
                  <CardDescription>
                    Manage automated data synchronization schedules
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-4 border rounded-lg">
                        <div>
                          <h4 className="font-medium">Patient Records</h4>
                          <p className="text-sm text-muted-foreground">
                            Every 4 hours
                          </p>
                        </div>
                        <Button variant="outline" size="sm">
                          <Edit className="h-4 w-4" />
                        </Button>
                      </div>
                      <div className="flex items-center justify-between p-4 border rounded-lg">
                        <div>
                          <h4 className="font-medium">Appointments</h4>
                          <p className="text-sm text-muted-foreground">
                            Every 30 minutes
                          </p>
                        </div>
                        <Button variant="outline" size="sm">
                          <Edit className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-4 border rounded-lg">
                        <div>
                          <h4 className="font-medium">Lab Results</h4>
                          <p className="text-sm text-muted-foreground">
                            Every 2 hours
                          </p>
                        </div>
                        <Button variant="outline" size="sm">
                          <Edit className="h-4 w-4" />
                        </Button>
                      </div>
                      <div className="flex items-center justify-between p-4 border rounded-lg">
                        <div>
                          <h4 className="font-medium">Medications</h4>
                          <p className="text-sm text-muted-foreground">
                            Daily at 2:00 AM
                          </p>
                        </div>
                        <Button variant="outline" size="sm">
                          <Edit className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Compliance Tab */}
            <TabsContent value="compliance" className="space-y-6">
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-healthcare-text">
                  Compliance Monitoring
                </h2>
                <Button variant="outline">
                  <Download className="mr-2 h-4 w-4" />
                  Export Audit Report
                </Button>
              </div>

              {/* Compliance Overview */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="bg-green-100 p-2 rounded-full">
                        <Shield className="h-5 w-5 text-green-600" />
                      </div>
                      <h3 className="font-semibold">HIPAA Compliance</h3>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">Encryption</span>
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Access Controls</span>
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Audit Logging</span>
                        <AlertTriangle className="h-4 w-4 text-yellow-600" />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="bg-blue-100 p-2 rounded-full">
                        <Activity className="h-5 w-5 text-blue-600" />
                      </div>
                      <h3 className="font-semibold">HL7 FHIR</h3>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">R4 Compliance</span>
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Data Validation</span>
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Schema Validation</span>
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="bg-purple-100 p-2 rounded-full">
                        <Key className="h-5 w-5 text-purple-600" />
                      </div>
                      <h3 className="font-semibold">API Security</h3>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">OAuth 2.0</span>
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Rate Limiting</span>
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Key Rotation</span>
                        <Clock className="h-4 w-4 text-gray-600" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Active Alerts */}
              <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 text-orange-600" />
                    Active Compliance Alerts
                  </CardTitle>
                  <CardDescription>
                    Critical security and compliance issues requiring attention
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {complianceAlerts
                      .filter((alert) => !alert.resolved)
                      .map((alert) => (
                        <div
                          key={alert.id}
                          className="flex items-center justify-between p-4 border rounded-lg"
                        >
                          <div className="flex items-center gap-4">
                            <div className="flex items-center gap-2">
                              {getSeverityBadge(alert.severity)}
                              <Badge variant="outline">{alert.type}</Badge>
                            </div>
                            <div>
                              <p className="font-medium">{alert.message}</p>
                              <p className="text-sm text-muted-foreground">
                                {formatTimestamp(alert.timestamp)}
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Button variant="outline" size="sm">
                              <Eye className="h-4 w-4" />
                            </Button>
                            <Button
                              size="sm"
                              className="bg-medical-green hover:bg-medical-green/90"
                            >
                              Resolve
                            </Button>
                          </div>
                        </div>
                      ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Analytics Tab */}
            <TabsContent value="analytics" className="space-y-6">
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-healthcare-text">
                  Integration Analytics
                </h2>
                <div className="flex gap-2">
                  <Select defaultValue="7d">
                    <SelectTrigger className="w-32">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="24h">Last 24h</SelectItem>
                      <SelectItem value="7d">Last 7 days</SelectItem>
                      <SelectItem value="30d">Last 30 days</SelectItem>
                      <SelectItem value="90d">Last 90 days</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button variant="outline">
                    <Download className="mr-2 h-4 w-4" />
                    Export
                  </Button>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">
                          Sync Success Rate
                        </p>
                        <p className="text-2xl font-bold text-green-600">
                          98.7%
                        </p>
                      </div>
                      <TrendingUp className="h-8 w-8 text-green-600" />
                    </div>
                    <Progress value={98.7} className="mt-3" />
                  </CardContent>
                </Card>

                <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">
                          Avg Response Time
                        </p>
                        <p className="text-2xl font-bold text-blue-600">
                          245ms
                        </p>
                      </div>
                      <Zap className="h-8 w-8 text-blue-600" />
                    </div>
                    <p className="text-xs text-green-600 mt-2">
                      ↓ 12% from last week
                    </p>
                  </CardContent>
                </Card>

                <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">
                          Daily API Calls
                        </p>
                        <p className="text-2xl font-bold text-purple-600">
                          24.3K
                        </p>
                      </div>
                      <BarChart3 className="h-8 w-8 text-purple-600" />
                    </div>
                    <p className="text-xs text-green-600 mt-2">
                      ↑ 8% from yesterday
                    </p>
                  </CardContent>
                </Card>

                <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">
                          Error Rate
                        </p>
                        <p className="text-2xl font-bold text-red-600">0.3%</p>
                      </div>
                      <AlertCircle className="h-8 w-8 text-red-600" />
                    </div>
                    <p className="text-xs text-green-600 mt-2">
                      ↓ 45% from last week
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Data Transfer Analytics */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle>Data Transfer Volume</CardTitle>
                    <CardDescription>
                      Records synchronized per day
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
                      <p className="text-muted-foreground">
                        Chart placeholder - Daily sync volume trends
                      </p>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle>System Performance</CardTitle>
                    <CardDescription>
                      Response times and uptime metrics
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
                      <p className="text-muted-foreground">
                        Chart placeholder - Performance metrics over time
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Top Performing Systems */}
              <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle>System Performance Ranking</CardTitle>
                  <CardDescription>
                    Based on uptime, response time, and error rates
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {ehrSystems
                      .sort((a, b) => b.uptime - a.uptime)
                      .map((system, index) => (
                        <div
                          key={system.id}
                          className="flex items-center justify-between p-4 border rounded-lg"
                        >
                          <div className="flex items-center gap-4">
                            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-medical-blue/10 text-medical-blue font-bold">
                              {index + 1}
                            </div>
                            <div>
                              <p className="font-medium">{system.name}</p>
                              <p className="text-sm text-muted-foreground">
                                {system.recordsCount.toLocaleString()} records
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-4">
                            <div className="text-right">
                              <p className="font-medium">
                                {system.uptime}% uptime
                              </p>
                              <p className="text-sm text-muted-foreground">
                                Last 30 days
                              </p>
                            </div>
                            {getStatusBadge(system.status)}
                          </div>
                        </div>
                      ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
