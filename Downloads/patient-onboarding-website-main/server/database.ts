import sqlite3 from 'sqlite3';
import { promisify } from 'util';

export interface Database {
  run: (sql: string, ...params: any[]) => Promise<sqlite3.RunResult>;
  get: (sql: string, ...params: any[]) => Promise<any>;
  all: (sql: string, ...params: any[]) => Promise<any[]>;
  close: () => Promise<void>;
}

class DatabaseManager {
  private db: sqlite3.Database | null = null;

  async initialize(): Promise<Database> {
    return new Promise((resolve, reject) => {
      this.db = new sqlite3.Database('healthcare.db', (err) => {
        if (err) {
          reject(err);
          return;
        }
        
        console.log('Connected to SQLite database');
        this.createTables().then(() => {
          resolve(this.createDatabaseInterface());
        }).catch(reject);
      });
    });
  }

  private async createTables(): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    const tables = [
      // Users table
      `CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        phone TEXT NOT NULL,
        date_of_birth TEXT NOT NULL,
        gender TEXT NOT NULL,
        blood_type TEXT,
        allergies TEXT,
        insurance_provider TEXT,
        policy_number TEXT,
        group_number TEXT,
        role TEXT DEFAULT 'patient',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,

      // Medical history table
      `CREATE TABLE IF NOT EXISTS medical_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        condition_name TEXT NOT NULL,
        diagnosed_year TEXT,
        status TEXT NOT NULL, -- 'current', 'past', 'family-history'
        notes TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
      )`,

      // Current medications table
      `CREATE TABLE IF NOT EXISTS current_medications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        medication_name TEXT NOT NULL,
        dosage TEXT,
        frequency TEXT,
        prescribed_by TEXT,
        start_date TEXT,
        notes TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
      )`,

      // Emergency contacts table
      `CREATE TABLE IF NOT EXISTS emergency_contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        relationship TEXT NOT NULL,
        primary_phone TEXT NOT NULL,
        secondary_phone TEXT,
        email TEXT,
        street_address TEXT,
        city TEXT,
        state TEXT,
        zip_code TEXT,
        is_authorized_to_receive_info BOOLEAN DEFAULT FALSE,
        can_make_healthcare_decisions BOOLEAN DEFAULT FALSE,
        notes TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
      )`,

      // Appointments table
      `CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        appointment_date TEXT NOT NULL,
        appointment_time TEXT NOT NULL,
        doctor_name TEXT NOT NULL,
        appointment_type TEXT NOT NULL,
        status TEXT DEFAULT 'scheduled', -- 'scheduled', 'completed', 'cancelled'
        location TEXT,
        notes TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
      )`,

      // Surgery history table
      `CREATE TABLE IF NOT EXISTS surgery_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        procedure_name TEXT NOT NULL,
        surgery_date TEXT NOT NULL,
        hospital TEXT,
        surgeon TEXT,
        complications TEXT,
        notes TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
      )`,

      // Family history table
      `CREATE TABLE IF NOT EXISTS family_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        relation TEXT NOT NULL,
        medical_conditions TEXT, -- JSON array of conditions
        age_at_death TEXT,
        cause_of_death TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
      )`,

      // Lifestyle information table
      `CREATE TABLE IF NOT EXISTS lifestyle_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        smoking_status TEXT,
        smoking_details TEXT,
        alcohol_consumption TEXT,
        alcohol_details TEXT,
        exercise_frequency TEXT,
        exercise_details TEXT,
        diet_type TEXT,
        diet_details TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
      )`,

      // Additional health information table
      `CREATE TABLE IF NOT EXISTS additional_health_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        hospitalizations TEXT,
        emergency_room_visits TEXT,
        significant_injuries TEXT,
        other_concerns TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
      )`
    ];

    for (const table of tables) {
      await this.runQuery(table);
    }

    console.log('Database tables created successfully');
  }

  private runQuery(sql: string, ...params: any[]): Promise<sqlite3.RunResult> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      
      this.db.run(sql, params, function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this);
        }
      });
    });
  }

  private getQuery(sql: string, ...params: any[]): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      
      this.db.get(sql, params, (err, row) => {
        if (err) {
          reject(err);
        } else {
          resolve(row);
        }
      });
    });
  }

  private allQuery(sql: string, ...params: any[]): Promise<any[]> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      
      this.db.all(sql, params, (err, rows) => {
        if (err) {
          reject(err);
        } else {
          resolve(rows);
        }
      });
    });
  }

  private createDatabaseInterface(): Database {
    return {
      run: this.runQuery.bind(this),
      get: this.getQuery.bind(this),
      all: this.allQuery.bind(this),
      close: () => {
        return new Promise((resolve, reject) => {
          if (this.db) {
            this.db.close((err) => {
              if (err) {
                reject(err);
              } else {
                console.log('Database connection closed');
                resolve();
              }
            });
          } else {
            resolve();
          }
        });
      }
    };
  }
}

// Global database instance
let dbInstance: Database | null = null;

export async function getDatabase(): Promise<Database> {
  if (!dbInstance) {
    const manager = new DatabaseManager();
    dbInstance = await manager.initialize();
  }
  return dbInstance;
}

export async function closeDatabase(): Promise<void> {
  if (dbInstance) {
    await dbInstance.close();
    dbInstance = null;
  }
}
